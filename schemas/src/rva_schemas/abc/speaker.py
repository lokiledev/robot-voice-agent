from collections.abc import Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

from rva_schemas.vision import ActiveSpeaker, BoundingBox


class SpeakerDetector(Protocol):
    """Who among the faces in view speaks, from their lips and the microphone together.

    Where the camera's and the microphone's branches meet: each feeds it from a thread of its own,
    with times from `time.monotonic`.
    """

    def hear(self, time: float, samples: npt.NDArray[np.float32]) -> None:
        """Take the microphone's next 16 kHz mono samples, the last of them heard at `time`."""
        ...

    def see(
        self, time: float, frame: npt.NDArray[np.uint8], faces: Mapping[int, BoundingBox]
    ) -> dict[int, ActiveSpeaker]:
        """Take a camera frame, taken at `time`, and the box of each face in it by track id.

        Returns:
            Whether each face speaks; faces are left out until enough sound is heard.
        """
        ...
