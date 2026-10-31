from collections import deque
from collections.abc import Callable, Sequence
from typing import Final

from rva_schemas.vision import BoundingBox, FaceAppeared, FaceEvent, FaceLeft, TrackedFace
from rva_vision_faces.config import TrackerConfig
from rva_vision_faces.detector import Detection
from rva_vision_faces.embedder import Embedding
from rva_vision_faces.gallery import Gallery, Match


class Track:
    """One face followed across frames. Owned and changed by the `Tracker` only."""

    def __init__(self, track_id: int, box: BoundingBox, now: float, samples: int) -> None:
        self.track_id: Final = track_id
        self.box = box
        self.first_seen: Final = now
        self.last_seen = now
        self.last_embedded = now
        self.confirmed = False
        self.embeddings: Final = deque[Embedding](maxlen=samples)
        self.match: Match | None = None


class Tracker:
    """Follows faces from frame to frame and says when they appear and leave.

    Detections continue a track when they overlap its last box. A detection that overlaps none is
    compared by embedding with the tracks that did not find their face in this frame, so that a
    face keeps its track across a short occlusion or a fast move.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config: Final = config
        self._tracks: Final[list[Track]] = []
        self._next_id = 1
        self._seen_now: set[int] = set()

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
    ) -> list[FaceEvent]:
        """Advance to a new frame.

        Args:
            now: Time of the frame, in seconds on a monotonic clock.
            detections: Faces large enough to track.
            embed: Embedding of a detected face; called only when a sample is due.
            gallery: Known people, to recognize the tracks.
        """
        matched = self._match_by_overlap(detections)
        for index, detection in enumerate(detections):
            if index in matched:
                track = matched[index]
                if now - track.last_embedded >= self._config.embed_every_s:
                    track.embeddings.append(embed(detection))
                    track.last_embedded = now
            else:
                embedding = embed(detection)
                track = self._reidentify(embedding, set(matched.values())) or self._new_track(
                    detection, now
                )
                track.embeddings.append(embedding)
                track.last_embedded = now
                matched[index] = track
            track.box = detection.box
            track.last_seen = now
        self._seen_now = {track.track_id for track in matched.values()}
        # Announced faces are allowed to flicker; new ones must be seen on every frame.
        self._tracks[:] = [
            track for track in self._tracks if track.confirmed or track.track_id in self._seen_now
        ]
        assignments = gallery.assign({track.track_id: track.embeddings for track in self._tracks})
        for track in self._tracks:
            track.match = assignments[track.track_id]
        return [*self._confirm(now), *self._leave(now)]

    def faces(self) -> tuple[TrackedFace, ...]:
        """The announced faces, including those briefly out of sight."""
        nearest = self._nearest()
        return tuple(
            _tracked_face(track, nearest=track is nearest)
            for track in self._tracks
            if track.confirmed
        )

    def find(self, track_id: int) -> Track | None:
        """An announced track by id."""
        return next(
            (track for track in self._tracks if track.track_id == track_id and track.confirmed),
            None,
        )

    def _match_by_overlap(self, detections: Sequence[Detection]) -> dict[int, Track]:
        pairs = sorted(
            (
                (_iou(track.box, detection.box), index, track)
                for track in self._tracks
                for index, detection in enumerate(detections)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matched: dict[int, Track] = {}
        for iou, index, track in pairs:
            if iou < self._config.iou_threshold:
                break
            if index not in matched and track not in matched.values():
                matched[index] = track
        return matched

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

    def _new_track(self, detection: Detection, now: float) -> Track:
        track = Track(self._next_id, detection.box, now, self._config.embeddings_per_track)
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


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    width = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    height = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    intersection = max(width, 0.0) * max(height, 0.0)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0
