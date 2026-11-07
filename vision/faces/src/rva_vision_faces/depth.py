from typing import Final

import cv2
import numpy as np
import onnxruntime

from rva_schemas.vision import CameraIntrinsics, DepthMap
from rva_vision_faces.detector import Image

_MEAN: Final = np.array([0.485, 0.456, 0.406], np.float32)
_STD: Final = np.array([0.229, 0.224, 0.225], np.float32)
_PAD: Final = (124, 116, 104)
"""ImageNet's mean in BGR: the padding is zero once normalized."""
_CANONICAL_FOCAL: Final = 300.0
"""Focal length, in pixels at the network's input, of the camera the depth is predicted for."""


class DepthEstimator:
    """Metric depth from a single image, with Depth Anything 3 Metric.

    The network predicts the depth a camera of a fixed focal length would see; the camera's own
    focal length, from its intrinsics, scales it to metres. Without them, the depth is left as
    predicted. The export takes a fixed square, so the frame is letterboxed into it: its pixels
    stay square, with one focal length for both axes.
    """

    def __init__(
        self, session: onnxruntime.InferenceSession, intrinsics: CameraIntrinsics | None
    ) -> None:
        _, _, height, width = session.get_inputs()[0].shape
        assert isinstance(height, int) and height == width, f"Expected a square input: {height}"
        self._session: Final = session
        self._size: Final = height
        self._intrinsics: Final = intrinsics

    def estimate(self, frame: Image) -> DepthMap:
        """Depth of a BGR frame, at the network's resolution with the frame's aspect ratio."""
        height, width = frame.shape[:2]
        square, rows = letterbox(frame, self._size)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        image = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        (depth,) = self._session.run(["depth"], {"image": np.ascontiguousarray(image)})
        metres = np.asarray(depth, np.float32)[0, 0, rows]
        if self._intrinsics is None:
            return DepthMap(metres=metres, calibrated=False)
        scale = metres_per_unit(self._intrinsics.scaled_to(width, height), self._size)
        return DepthMap(metres=metres * np.float32(scale), calibrated=self._intrinsics.calibrated)


def letterbox(frame: Image, size: int) -> tuple[Image, slice]:
    """Scale a landscape frame to `size` wide and pad it into a square, above and below.

    Returns:
        The square, and its rows that hold the frame.
    """
    height, width = frame.shape[:2]
    assert width >= height, f"Expected a landscape frame: {width}x{height}"
    rows = round(height * size / width)
    top = (size - rows) // 2
    scaled = cv2.resize(frame, (size, rows), interpolation=cv2.INTER_AREA)
    square = cv2.copyMakeBorder(
        scaled, top, size - rows - top, 0, 0, cv2.BORDER_CONSTANT, value=_PAD
    )
    return np.asarray(square, np.uint8), slice(top, top + rows)


def metres_per_unit(intrinsics: CameraIntrinsics, size: int) -> float:
    """Metres per unit of the network's output, for a frame letterboxed into `size` squared."""
    focal = (intrinsics.fx + intrinsics.fy) / 2 * size / intrinsics.width
    return focal / _CANONICAL_FOCAL
