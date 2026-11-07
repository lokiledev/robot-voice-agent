import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final

from rva_schemas.vision import ActiveSpeaker, BoundingBox, FaceFeatures, MainUser
from rva_vision_faces.config import EyeContactConfig, LipMotionConfig

_STICKINESS: Final = 1.25
"""How much better another face must score to take over as the main user: two people about as
close and central do not swap at every frame."""

# Face mesh points: the middle of the inner lips, then the inner corners of the mouth.
_UPPER_LIP: Final = 13
_LOWER_LIP: Final = 14
_MOUTH_CORNERS: Final = (78, 308)


class Attention:
    """Follows the main user, their eye contact and whether their lips move.

    The main user is the closest and most central face. Fed every frame, in the frame's order.
    """

    def __init__(self, eye_contact: EyeContactConfig, lip_motion: LipMotionConfig) -> None:
        self._eye_contact_config: Final = eye_contact
        self._lip_motion_config: Final = lip_motion
        self._track_id: int | None = None
        self._angles: Final = _Window(eye_contact.window_s)
        """How far off the camera the main user's gaze was."""
        self._apertures: Final = _Window(lip_motion.window_s)
        """How far apart the main user's lips were."""
        self._eye_contact = False
        self._lips_moving = False

    def update(
        self,
        now: float,
        faces: Mapping[int, tuple[BoundingBox, FaceFeatures]],
        speakers: Mapping[int, ActiveSpeaker],
        width: int,
        height: int,
    ) -> MainUser | None:
        """Take the faces seen in a frame, by track id, and return the main user among them.

        Args:
            now: When the frame was taken, in seconds.
            faces: The box and features of each face seen in the frame.
            speakers: Whether each face speaks, from its lips and the sound; empty without a
                microphone.
            width: The frame's width, in pixels.
            height: The frame's height, in pixels.

        Returns:
            `None` when no face is in view.
        """
        track_id = self._pick(faces, width, height)
        if track_id != self._track_id:
            self._track_id = track_id
            self._angles.clear()
            self._apertures.clear()
            self._eye_contact = False
            self._lips_moving = False
        if track_id is None:
            return None
        features = faces[track_id][1]
        angle = statistics.median(self._angles.add(now, features.gaze.off_camera()))
        eye_contact = self._eye_contact_config
        self._eye_contact = angle < (
            eye_contact.leave_deg if self._eye_contact else eye_contact.enter_deg
        )
        motion = statistics.pstdev(self._apertures.add(now, lip_aperture(features.landmarks)))
        lip_motion = self._lip_motion_config
        self._lips_moving = motion > (lip_motion.leave if self._lips_moving else lip_motion.enter)
        return MainUser(
            track_id=track_id,
            gaze_off_camera=angle,
            eye_contact=self._eye_contact,
            mouth_motion=motion,
            lips_moving=self._lips_moving,
            active_speaker=speakers.get(track_id),
        )

    def _pick(
        self, faces: Mapping[int, tuple[BoundingBox, FaceFeatures]], width: int, height: int
    ) -> int | None:
        scores = {track_id: prominence(box, width, height) for track_id, (box, _) in faces.items()}
        if not scores:
            return None
        best = max(scores, key=scores.__getitem__)
        current = self._track_id
        if current in scores and scores[best] < _STICKINESS * scores[current]:
            return current
        return best


class _Window:
    """The values of the last few seconds."""

    def __init__(self, seconds: float) -> None:
        self._seconds: Final = seconds
        self._values: Final = deque[tuple[float, float]]()

    def add(self, now: float, value: float) -> list[float]:
        """Add a value and return those of the window, oldest first: at least this one."""
        self._values.append((now, value))
        while self._values[0][0] < now - self._seconds:
            self._values.popleft()
        return [value for _, value in self._values]

    def clear(self) -> None:
        self._values.clear()


def prominence(box: BoundingBox, width: int, height: int) -> float:
    """How much a face is in front of the robot, from its size and how central it is.

    Its height as a fraction of the frame's, less the farther it is from the middle, down to
    nothing at the edges.
    """
    offset = abs(box.x + box.width / 2 - width / 2) / (width / 2)
    return box.height / height * max(1 - offset, 0.0)


def lip_aperture(landmarks: Sequence[tuple[float, float]]) -> float:
    """Gap between the inner lips over the mouth's width: 0 closed, about 0.5 wide open.

    The face mesh's lips follow speech better than the jaw-open blendshape, which barely moves
    when only the lips do.
    """
    left, right = (landmarks[index] for index in _MOUTH_CORNERS)
    return math.dist(landmarks[_UPPER_LIP], landmarks[_LOWER_LIP]) / math.dist(left, right)
