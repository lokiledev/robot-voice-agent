"""Where the robot's head camera was, from the head controller, for the perception."""

from collections import deque
from typing import Final

from rva_robot_client.head import CameraPoseReader
from rva_schemas.frames import CameraPose

_KEEP_S: Final = 2.0
"""How long the camera's past poses are kept: longer than a frame takes to be processed."""
_MAX_GAP_S: Final = 0.05
"""How far from a frame's time the nearest pose may be: five joint states at 100 Hz."""


class PublishedCameraPoses:
    """`CameraPoses` from the head controller's topic, as the robot's joints place the camera.

    Not thread-safe: use it from the thread that processes the frames.
    """

    def __init__(self, reader: CameraPoseReader) -> None:
        self._reader: Final = reader
        self._poses: Final = deque[tuple[float, CameraPose]]()

    def at(self, time: float) -> CameraPose | None:
        for stamped in self._reader.received():
            pose = CameraPose(position=stamped.position, rotation=stamped.rotation)
            self._poses.append((stamped.timestamp, pose))
        while self._poses and self._poses[0][0] < self._poses[-1][0] - _KEEP_S:
            self._poses.popleft()
        nearest = min(self._poses, key=lambda noted: abs(noted[0] - time), default=None)
        if nearest is None or abs(nearest[0] - time) > _MAX_GAP_S:
            return None
        return nearest[1]
