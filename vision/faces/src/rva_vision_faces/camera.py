from typing import Final

import cv2
import numpy as np

from rva_vision_faces.config import CameraConfig
from rva_vision_faces.detector import Image


class Camera:
    """Frames from a V4L2 camera, in MJPG so that high resolutions keep their frame rate."""

    def __init__(self, config: CameraConfig) -> None:
        self._config: Final = config
        self._capture: Final = cv2.VideoCapture(str(config.device), cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera {config.device}")
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        # Process the latest frame rather than one queued while the previous was processed.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if size != (config.width, config.height):
            self._capture.release()
            raise RuntimeError(
                f"Camera {config.device} gives {size[0]}x{size[1]}, not {config.width}x{config.height}"
            )

    def read(self) -> Image:
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError(f"Camera {self._config.device} stopped giving frames")
        return np.asarray(frame, np.uint8)

    def close(self) -> None:
        self._capture.release()
