import math
from collections import deque
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from rva_schemas.frames import CameraPose
from rva_schemas.vision import (
    BoundingBox,
    CameraIntrinsics,
    FaceAppeared,
    FaceEvent,
    FaceLeft,
    TrackedFace,
)
from rva_vision_faces.config import TrackerConfig
from rva_vision_faces.detector import Detection
from rva_vision_faces.embedder import Embedding
from rva_vision_faces.gallery import Gallery, Match


class Viewpoint(NamedTuple):
    """Where a turning camera was when it took a frame, and how it sees."""

    pose: CameraPose
    intrinsics: CameraIntrinsics


class Track:
    """One face followed across frames. Owned and changed by the `Tracker` only."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        seen_from: Viewpoint | None,
        now: float,
        samples: int,
    ) -> None:
        self.track_id: Final = track_id
        self.detection = detection
        """The face's last detection."""
        self.seen_from = seen_from
        """Where the camera was at the face's last detection; `None` for a camera that does not
        turn, or not known to then."""
        self.expected = detection.box
        """Where the face would be in the latest frame, had it not moved: its last box, moved
        with the image as the camera turned since."""
        self.first_seen: Final = now
        self.last_seen = now
        self.last_embedded = now
        self.confirmed = False
        self.embeddings: Final = deque[Embedding](maxlen=samples)
        self.match: Match | None = None

    @property
    def box(self) -> BoundingBox:
        return self.detection.box


class Tracker:
    """Follows faces from frame to frame and says when they appear and leave.

    Detections continue a track when they overlap its last box, or else when their centre is
    close to it for the face's size and the time since the last frame: a small face far away
    overlaps little after a quick turn.
    Where the camera turns, each last box is first moved by its turn since that box was seen, so
    the head's motion does not tear the tracks: since then, not since the last frame, so a frame
    whose pose is not known, or a face missed while the head turned, loses nothing. A detection
    that neither overlaps nor is close to any is compared by embedding with the tracks that did
    not find their face in this frame, so that a face keeps its track across a short occlusion or
    a fast move.

    Once a track is recognized it keeps its person for as long as it lives, and its face is no
    longer embedded: a track that stays continuous is the same face. Unrecognized tracks are
    embedded every `embed_every_s` and matched again with each new sample.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config: Final = config
        self._tracks: Final[list[Track]] = []
        self._next_id = 1
        self._seen_now: set[int] = set()
        self._updated_at: float | None = None

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Every track, announced or not."""
        return tuple(self._tracks)

    def update(
        self,
        now: float,
        detections: Sequence[Detection],
        embed: Callable[[Detection], Embedding],
        gallery: Gallery,
        viewpoint: Viewpoint | None,
    ) -> list[FaceEvent]:
        """Advance to a new frame.

        Args:
            now: Time of the frame, in seconds on a monotonic clock.
            detections: Faces large enough to track.
            embed: Embedding of a detected face; called only when a sample is due.
            gallery: Known people, to recognize the tracks.
            viewpoint: Where the camera was at the frame; `None` for a camera that does not
                turn, or not known to.
        """
        for track in self._tracks:
            track.expected = _expected(track, viewpoint)
        elapsed = 0.0 if self._updated_at is None else now - self._updated_at
        self._updated_at = now
        matched = self._match_by_overlap(detections)
        matched |= self._match_by_distance(detections, matched, self._config.max_speed * elapsed)
        sampled: list[Track] = []
        for index, detection in enumerate(detections):
            if index in matched:
                track = matched[index]
                if track.match is None and now - track.last_embedded >= self._config.embed_every_s:
                    track.embeddings.append(embed(detection))
                    track.last_embedded = now
                    sampled.append(track)
            else:
                embedding = embed(detection)
                track = self._reidentify(embedding, set(matched.values())) or self._new_track(
                    detection, viewpoint, now
                )
                track.embeddings.append(embedding)
                track.last_embedded = now
                sampled.append(track)
                matched[index] = track
            track.detection = detection
            track.seen_from = viewpoint
            track.expected = detection.box
            track.last_seen = now
        self._seen_now = {track.track_id for track in matched.values()}
        # Announced faces are allowed to flicker; new ones must be seen on every frame.
        self._tracks[:] = [
            track for track in self._tracks if track.confirmed or track.track_id in self._seen_now
        ]
        self._recognize([track for track in sampled if track.match is None], gallery)
        return [*self._confirm(now), *self._leave(now)]

    def recognize(self, gallery: Gallery) -> None:
        """Match every track again, once the known people changed."""
        for track in self._tracks:
            track.match = None
        self._recognize(self._tracks, gallery)

    def faces(self) -> tuple[TrackedFace, ...]:
        """The announced faces, including those briefly out of sight."""
        nearest = self._nearest()
        return tuple(
            _tracked_face(track, nearest=track is nearest)
            for track in self._tracks
            if track.confirmed
        )

    def visible(self) -> tuple[Track, ...]:
        """The tracks whose face was detected in the last frame, announced or not."""
        return tuple(track for track in self._tracks if track.track_id in self._seen_now)

    def find(self, track_id: int) -> Track | None:
        """An announced track by id."""
        return next(
            (track for track in self._tracks if track.track_id == track_id and track.confirmed),
            None,
        )

    def _match_by_overlap(self, detections: Sequence[Detection]) -> dict[int, Track]:
        pairs = sorted(
            (
                (iou(track.expected, detection.box), index, track)
                for track in self._tracks
                for index, detection in enumerate(detections)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matched: dict[int, Track] = {}
        for overlap, index, track in pairs:
            if overlap < self._config.iou_threshold:
                break
            if index not in matched and track not in matched.values():
                matched[index] = track
        return matched

    def _match_by_distance(
        self, detections: Sequence[Detection], matched: dict[int, Track], max_shift: float
    ) -> dict[int, Track]:
        """The detections left, each with the closest track left within `max_shift` of it.

        Args:
            detections: Of this frame.
            matched: Those that overlap a track, which are left out with their tracks.
            max_shift: How far a face's centre may have moved since the last frame, in its
                heights.
        """
        taken = set(matched.values())
        pairs = sorted(
            (
                (_shift(track.expected, detection.box), index, track)
                for track in self._tracks
                if track not in taken
                for index, detection in enumerate(detections)
                if index not in matched
            ),
            key=lambda pair: pair[0],
        )
        found: dict[int, Track] = {}
        for shift, index, track in pairs:
            if shift > max_shift:
                break
            if index not in found and track not in found.values():
                found[index] = track
        return found

    def _recognize(self, tracks: Sequence[Track], gallery: Gallery) -> None:
        """Match unrecognized tracks; a person already on another track stays there."""
        taken = {track.match.person for track in self._tracks if track.match is not None}
        assignments = gallery.assign({track.track_id: track.embeddings for track in tracks})
        for track in tracks:
            match = assignments[track.track_id]
            if match is not None and match.person not in taken:
                track.match = match

    def _reidentify(self, embedding: Embedding, taken: set[Track]) -> Track | None:
        candidates = [
            (max(float(embedding @ sample) for sample in track.embeddings), track)
            for track in self._tracks
            if track.confirmed and track not in taken and track.embeddings
        ]
        if not candidates:
            return None
        score, track = max(candidates, key=lambda candidate: candidate[0])
        return track if score >= self._config.reid_threshold else None

    def _new_track(self, detection: Detection, viewpoint: Viewpoint | None, now: float) -> Track:
        track = Track(self._next_id, detection, viewpoint, now, self._config.embeddings_per_track)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _confirm(self, now: float) -> list[FaceEvent]:
        due = [
            track
            for track in self._tracks
            if not track.confirmed and now - track.first_seen >= self._config.confirm_after_s
        ]
        for track in due:
            track.confirmed = True
        nearest = self._nearest()
        return [FaceAppeared(face=_tracked_face(track, nearest=track is nearest)) for track in due]

    def _leave(self, now: float) -> list[FaceEvent]:
        gone = [
            track for track in self._tracks if now - track.last_seen >= self._config.leave_after_s
        ]
        for track in gone:
            self._tracks.remove(track)
        return [
            FaceLeft(
                track_id=track.track_id, person=None if track.match is None else track.match.person
            )
            for track in gone
        ]

    def _nearest(self) -> Track | None:
        visible = [
            track for track in self._tracks if track.confirmed and track.track_id in self._seen_now
        ]
        if not visible:
            return None
        return max(visible, key=lambda track: track.box.height)


def _tracked_face(track: Track, *, nearest: bool) -> TrackedFace:
    return TrackedFace(
        track_id=track.track_id,
        box=track.box,
        person=None if track.match is None else track.match.person,
        nearest=nearest,
    )


def image_turn(
    earlier: CameraPose, now: CameraPose, intrinsics: CameraIntrinsics
) -> npt.NDArray[np.float64]:
    """How the image moves with the camera's turn from `earlier` to `now`, as a homography.

    It takes a pixel of the frame taken at `earlier` to where the same direction is seen at
    `now`: the turn alone, the camera's few centimetres of travel left out.
    """
    camera = np.array(
        ((intrinsics.fx, 0.0, intrinsics.cx), (0.0, intrinsics.fy, intrinsics.cy), (0.0, 0.0, 1.0))
    )
    return camera @ np.array(now.turn_since(earlier)) @ np.linalg.inv(camera)


def _expected(track: Track, viewpoint: Viewpoint | None) -> BoundingBox:
    """Where the track's face would be seen from `viewpoint`, had it not moved.

    Its last box as it is, without both viewpoints: the camera's turn between them is not known.
    """
    if viewpoint is None or track.seen_from is None:
        return track.box
    return _moved(track.box, image_turn(track.seen_from.pose, viewpoint.pose, viewpoint.intrinsics))


def _moved(box: BoundingBox, turn: npt.NDArray[np.float64]) -> BoundingBox:
    """`box` with its centre moved by the homography `turn`, its size kept."""
    x, y, w = turn @ np.array((box.x + box.width / 2, box.y + box.height / 2, 1.0))
    return box.model_copy(
        update={"x": float(x / w) - box.width / 2, "y": float(y / w) - box.height / 2}
    )


def _shift(a: BoundingBox, b: BoundingBox) -> float:
    """How far apart the centres of two boxes are, in heights of `a`."""
    dx = (b.x + b.width / 2) - (a.x + a.width / 2)
    dy = (b.y + b.height / 2) - (a.y + a.height / 2)
    return math.hypot(dx, dy) / a.height


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes, 0 when both are empty."""
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    intersection = max(width, 0.0) * max(height, 0.0)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0
