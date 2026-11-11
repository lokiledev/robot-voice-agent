import struct
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel

_HEADER: Final = struct.Struct("<dII")
"""Before the pixels: the timestamp, then the height and the width."""


class CameraFrame(RVABaseModel, arbitrary_types_allowed=True):
    timestamp: float
    """When the frame was taken, in seconds of `time.monotonic()`, the clock the joint states are
    stamped with."""
    image: npt.NDArray[np.uint8]
    """BGR, height × width × 3, as OpenCV reads a camera."""


def encoded_size(width: int, height: int) -> int:
    """Bytes a frame of this size takes encoded."""
    return _HEADER.size + width * height * 3


def encode_frame(frame: CameraFrame, buffer: memoryview) -> int:
    """Write `frame` at the start of `buffer`, without an intermediate copy.

    Returns:
        How many bytes were written.
    """
    height, width, _ = frame.image.shape
    _HEADER.pack_into(buffer, 0, frame.timestamp, height, width)
    end = encoded_size(width, height)
    buffer[_HEADER.size : end] = np.ascontiguousarray(frame.image).reshape(-1)
    return end


def decode_frame(buffer: memoryview) -> CameraFrame:
    """The frame `encode_frame` wrote, copied out of `buffer`, which may be reused after."""
    timestamp, height, width = _HEADER.unpack_from(buffer)
    pixels = np.frombuffer(buffer, np.uint8, count=height * width * 3, offset=_HEADER.size)
    return CameraFrame(timestamp=timestamp, image=pixels.reshape(height, width, 3).copy())
