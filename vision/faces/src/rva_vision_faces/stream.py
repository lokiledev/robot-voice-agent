import contextlib
import subprocess
from typing import Final

import cv2

from rva_vision_faces.config import VideoStreamConfig
from rva_vision_faces.detector import Image

_KEYFRAME_EVERY: Final = 30
"""Frames between keyframes: a page that starts playing waits for the next one."""
_STOP_TIMEOUT_S: Final = 5.0


class RtspPublisher:
    """Publishes frames over RTSP as H.264, through an ffmpeg process. Lives in one thread.

    The encoding suits WebRTC players: baseline profile, so no B-frames, and frequent keyframes.
    """

    def __init__(self, config: VideoStreamConfig, width: int, height: int) -> None:
        """Start ffmpeg, to publish frames of `width` by `height` pixels to `config.url`."""
        # H.264 in 4:2:0 needs even sizes.
        self._size: Final = (
            config.width // 2 * 2,
            round(height * config.width / width) // 2 * 2,
        )
        self._url: Final = config.url
        self._process: Final = subprocess.Popen(
            publish_command(config.url, *self._size), stdin=subprocess.PIPE
        )

    def send(self, frame: Image) -> None:
        """Send a frame, scaled down; blocks while ffmpeg catches up.

        Raises:
            RuntimeError: If ffmpeg stopped, e.g. because the RTSP server is not running.
        """
        small = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        assert self._process.stdin is not None, "started with a pipe"
        try:
            self._process.stdin.write(small.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(
                f"ffmpeg stopped publishing to {self._url}, exit code {self._process.wait()}"
            ) from error

    def close(self) -> None:
        assert self._process.stdin is not None, "started with a pipe"
        # Already gone if ffmpeg stopped, which `send` reported.
        with contextlib.suppress(BrokenPipeError):
            self._process.stdin.close()
        try:
            self._process.wait(_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


def publish_command(url: str, width: int, height: int) -> list[str]:
    """The ffmpeg command reading raw BGR frames on stdin and publishing them to `url`."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        # Each frame is stamped when it arrives, since the frame rate varies with the load.
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        "-",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(_KEYFRAME_EVERY),
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        url,
    ]
