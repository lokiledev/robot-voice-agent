from collections.abc import Sequence
from typing import Final

import cv2

from rva_pydantic import RVABaseModel
from rva_vision_faces.detector import Image
from rva_vision_faces.tracker import Track

_NAME: Final = "rva see"
# BGR
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (0, 165, 255)
_UNCONFIRMED: Final = (160, 160, 160)


class Timings(RVABaseModel):
    """Smoothed per-frame figures shown over the image."""

    fps: float
    capture_ms: float
    detect_ms: float
    embed_ms: float


class DebugWindow:
    """A resizable window showing the camera with each tracked face. Lives in one thread."""

    def __init__(self, width: int, height: int) -> None:
        cv2.namedWindow(_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_NAME, width, height)

    def show(self, frame: Image) -> str | None:
        """Display a frame and return the key pressed, or `"q"` once the window is closed."""
        cv2.imshow(_NAME, frame)
        key = cv2.waitKey(1)
        if cv2.getWindowProperty(_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "q"
        return chr(key & 0xFF) if key >= 0 else None

    def close(self) -> None:
        cv2.destroyWindow(_NAME)


def draw(frame: Image, tracks: Sequence[Track], timings: Timings, caption: str | None) -> None:
    """Draw each track's box and label, the timings and the user's last sentence onto the frame."""
    scale = frame.shape[0] / 540
    thickness = max(1, round(2 * scale))
    for track in tracks:
        x, y = round(track.box.x), round(track.box.y)
        right, bottom = round(track.box.x + track.box.width), round(track.box.y + track.box.height)
        color = _color(track)
        cv2.rectangle(frame, (x, y), (right, bottom), color, thickness)
        cv2.putText(
            frame,
            label(track),
            (x, max(y - round(8 * scale), round(20 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale,
            color,
            thickness,
        )
    overlay = (
        f"{timings.fps:.0f} fps  capture {timings.capture_ms:.1f} ms  "
        f"detect {timings.detect_ms:.1f} ms  embed {timings.embed_ms:.1f} ms"
    )
    cv2.putText(
        frame,
        overlay,
        (round(10 * scale), round(25 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        (255, 255, 255),
        thickness,
    )
    if caption is not None:
        _draw_caption(frame, caption, scale, thickness)


def label(track: Track) -> str:
    if track.match is None:
        return f"#{track.track_id} unknown"
    person = track.match.person
    return f"#{track.track_id} {person.name} ({str(person.user_id)[:8]}) {track.match.score:.2f}"


def _color(track: Track) -> tuple[int, int, int]:
    if not track.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if track.match is None else _KNOWN


def _draw_caption(frame: Image, caption: str, scale: float, thickness: int) -> None:
    text = f"User: {caption}"
    font_scale = 0.7 * scale
    (_, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bottom = frame.shape[0] - round(15 * scale)
    # A dark band keeps the text readable over any background.
    cv2.rectangle(
        frame,
        (0, bottom - text_height - round(12 * scale)),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (round(10 * scale), bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )
