import math
import statistics
from collections import deque
from collections.abc import Mapping
from typing import Final

from rva_schemas.vision import BoundingBox, Gaze, MainUser
from rva_vision_faces.config import EyeContactConfig

_STICKINESS: Final = 1.25
"""How much better another face must score to take over as the main user: two people about as
close and central do not swap at every frame."""


class Attention:
    """Follows the main user, the closest and most central face, and whether they make eye contact.

    Fed every frame, in the frame's order.
    """

    def __init__(self, config: EyeContactConfig) -> None:
        self._config: Final = config
        self._track_id: int | None = None
        self._angles: Final = deque[tuple[float, float]]()
        """When, and how far off the camera, the main user's gaze was."""
        self._eye_contact = False

    def update(
        self, now: float, faces: Mapping[int, tuple[BoundingBox, Gaze]], width: int, height: int
    ) -> MainUser | None:
        """Take the faces seen in a frame, by track id, and return the main user among them.

        Args:
            now: When the frame was taken, in seconds.
            faces: The box and gaze of each face seen in the frame.
            width: The frame's width, in pixels.
            height: The frame's height, in pixels.

        Returns:
            `None` when no face is in view.
        """
        track_id = self._pick(faces, width, height)
        if track_id != self._track_id:
            self._track_id = track_id
            self._angles.clear()
            self._eye_contact = False
        if track_id is None:
            return None
        self._angles.append((now, off_camera(faces[track_id][1])))
        while self._angles[0][0] < now - self._config.window_s:
            self._angles.popleft()
        angle = statistics.median(angle for _, angle in self._angles)
        threshold = self._config.leave_deg if self._eye_contact else self._config.enter_deg
        self._eye_contact = angle < threshold
        return MainUser(track_id=track_id, gaze_off_camera=angle, eye_contact=self._eye_contact)

    def _pick(
        self, faces: Mapping[int, tuple[BoundingBox, Gaze]], width: int, height: int
    ) -> int | None:
        scores = {track_id: prominence(box, width, height) for track_id, (box, _) in faces.items()}
        if not scores:
            return None
        best = max(scores, key=scores.__getitem__)
        current = self._track_id
        if current in scores and scores[best] < _STICKINESS * scores[current]:
            return current
        return best


def prominence(box: BoundingBox, width: int, height: int) -> float:
    """How much a face is in front of the robot, from its size and how central it is.

    Its height as a fraction of the frame's, less the farther it is from the middle, down to
    nothing at the edges.
    """
    offset = abs(box.x + box.width / 2 - width / 2) / (width / 2)
    return box.height / height * max(1 - offset, 0.0)


def off_camera(gaze: Gaze) -> float:
    """Angle between the gaze and the line to the camera, in degrees."""
    yaw, pitch = math.radians(gaze.yaw), math.radians(gaze.pitch)
    return math.degrees(math.acos(min(math.cos(yaw) * math.cos(pitch), 1.0)))
