from typing import Protocol

from rva_schemas.frames import CameraPose


class CameraPoses(Protocol):
    """Where a moving camera was, moment by moment: a robot's head camera."""

    def at(self, time: float) -> CameraPose | None:
        """Where the camera was at `time`; `None` when nothing tells it near then.

        `time` is in seconds of the monotonic clock frames are stamped with.
        """
        ...
