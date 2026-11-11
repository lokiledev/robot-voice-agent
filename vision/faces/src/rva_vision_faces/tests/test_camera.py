import threading
from pathlib import Path

import cv2
import numpy as np

from rva_vision_faces.camera import VideoFile, fit
from rva_vision_faces.config import CameraConfig, V4l2Source

CAMERA = CameraConfig(
    source=V4l2Source(device=Path("/dev/video0"), fps=30),
    frame="camera_optical",
    width=64,
    height=36,
    intrinsics=None,
)


def test_a_frame_of_another_shape_is_centered_on_black() -> None:
    square = np.full((10, 10, 3), 255, np.uint8)

    fitted = fit(square, (40, 20))

    assert fitted.shape == (20, 40, 3)
    # Scaled to 20 by 20, with 10 black columns on each side.
    assert fitted[:, :10].max() == 0
    assert fitted[:, 10:30].min() == 255
    assert fitted[:, 30:].max() == 0


def test_a_video_file_starts_with_its_first_frame_and_holds_its_last(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 50, (64, 36))
    # Three frames, each brighter than the one before.
    for level in (0, 100, 200):
        writer.write(np.full((36, 64, 3), level, np.uint8))
    writer.release()
    started = threading.Event()
    video = VideoFile(path, CAMERA, started)
    assert not started.is_set()

    first = video.read()

    assert started.is_set()
    assert first.shape == (36, 64, 3)
    assert first.mean() < 10
    # Whatever frames are skipped to keep the pace, the file ends on its last one and stays there.
    frames = [video.read() for _ in range(5)]
    assert abs(frames[-1].mean() - 200) < 10
    assert frames[-1] is frames[-2]
    video.close()
