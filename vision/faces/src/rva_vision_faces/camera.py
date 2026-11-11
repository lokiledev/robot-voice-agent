import threading
import time
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np

from rva_vision_faces.config import CameraConfig, V4l2Source
from rva_vision_faces.detector import Image


class FrameSource(Protocol):
    """Where the camera pipeline takes its frames from, in its own thread."""

    def read(self) -> Image:
        """The frame of the moment, BGR, blocking until there is one.

        Raises:
            RuntimeError: If the source stopped giving frames.
        """
        ...

    def close(self) -> None: ...


class Camera:
    """Frames from a V4L2 camera, in MJPG so that high resolutions keep their frame rate."""

    def __init__(self, source: V4l2Source, width: int, height: int) -> None:
        self._source: Final = source
        self._capture: Final = cv2.VideoCapture(str(source.device), cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera {source.device}")
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, source.fps)
        # Process the latest frame rather than one queued while the previous was processed.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if size != (width, height):
            self._capture.release()
            raise RuntimeError(
                f"Camera {source.device} gives {size[0]}x{size[1]}, not {width}x{height}"
            )

    def read(self) -> Image:
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError(f"Camera {self._source.device} stopped giving frames")
        return np.asarray(frame, np.uint8)

    def close(self) -> None:
        self._capture.release()


class VideoFile:
    """Frames from a video file at the pace they were filmed, in place of a camera.

    Like a camera, it gives the frame of the moment: frames that came due while the previous one
    was processed are skipped. Once the file ends, its last frame stays, so what listens to the
    file's sound can finish. Frames are fitted into the camera's size, with black bars around
    another shape, so everything sized for the camera holds.
    """

    def __init__(self, path: Path, config: CameraConfig, started: threading.Event) -> None:
        """Open the file; the film starts with the first frame read.

        Args:
            path: A video file OpenCV can decode.
            config: The camera stood in for, whose frame size is kept.
            started: Set as the first frame is read, for the file's sound to start with it.

        Raises:
            RuntimeError: If the file cannot be read as a video.
        """
        self._path: Final = path
        self._size: Final = (config.width, config.height)
        self._started: Final = started
        self._capture: Final = cv2.VideoCapture(str(path))
        self._fps: Final = self._capture.get(cv2.CAP_PROP_FPS)
        if not self._capture.isOpened() or self._fps <= 0:
            raise RuntimeError(f"Cannot read {path} as a video")
        self._start: float | None = None
        self._next = 0
        """Index of the next frame the file gives."""
        self._last: Image | None = None

    def read(self) -> Image:
        if self._start is None:
            self._start = time.monotonic()
            self._started.set()
        due = int((time.monotonic() - self._start) * self._fps)
        while self._next < due and self._capture.grab():
            self._next += 1
        ok, frame = self._capture.read()
        if not ok:
            if self._last is None:
                raise RuntimeError(f"{self._path} has no frame")
            time.sleep(1 / self._fps)
            return self._last
        time.sleep(max(self._start + self._next / self._fps - time.monotonic(), 0.0))
        self._next += 1
        self._last = fit(np.asarray(frame, np.uint8), self._size)
        return self._last

    def close(self) -> None:
        self._capture.release()


def fit(frame: Image, size: tuple[int, int]) -> Image:
    """`frame` scaled to fit in `size`, width by height, centered on black."""
    width, height = size
    scale = min(width / frame.shape[1], height / frame.shape[0])
    scaled = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    fitted = np.zeros((height, width, 3), np.uint8)
    top, left = (height - scaled.shape[0]) // 2, (width - scaled.shape[1]) // 2
    fitted[top : top + scaled.shape[0], left : left + scaled.shape[1]] = scaled
    return fitted
