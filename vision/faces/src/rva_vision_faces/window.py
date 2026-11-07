import textwrap
from collections.abc import Sequence
from typing import Final

import cv2
import numpy as np

from rva_pydantic import RVABaseModel
from rva_schemas.vision import BODY_CONNECTIONS, HAND_CONNECTIONS, Body, FaceFeatures
from rva_vision_faces.detector import Image
from rva_vision_faces.scene import Scene
from rva_vision_faces.tracker import Track

_NAME: Final = "lune see"
# BGR
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (0, 165, 255)
_UNCONFIRMED: Final = (160, 160, 160)
_SPEECH_LINE_CHARS: Final = 40
# Face mesh outlines, from MediaPipe's face_mesh_connections: both eyes, then the lips.
_FACE_CONTOURS: Final = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191),
)  # fmt: skip
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""


class Timings(RVABaseModel):
    """Smoothed per-frame figures shown over the image."""

    fps: float
    capture_ms: float
    detect_ms: float
    embed_ms: float
    faces_ms: float
    bodies_ms: float
    """Time left waiting for the skeletons once the faces are done: they run alongside."""


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


def draw(
    frame: Image,
    tracks: Sequence[Track],
    scene: Scene,
    timings: Timings,
    caption: tuple[int, str] | None,
) -> None:
    """Draw the tracks and what was seen of them, the timings, and the user's last sentence.

    Each track gets its box, label, face features and skeleton. `caption` is the track id of the face that said the sentence, and the sentence; it hangs under
    that face's box. Skeletons without a face are drawn in gray.
    """
    scale = frame.shape[0] / 540
    thickness = max(1, round(2 * scale))
    for body in scene.unmatched_bodies:
        _draw_body(frame, body, _UNCONFIRMED, thickness)
    for track in tracks:
        x, y = round(track.box.x), round(track.box.y)
        right, bottom = round(track.box.x + track.box.width), round(track.box.y + track.box.height)
        color = _color(track)
        if (body := scene.bodies.get(track.track_id)) is not None:
            _draw_body(frame, body, color, thickness)
        cv2.rectangle(frame, (x, y), (right, bottom), color, thickness)
        lines = [label(track)]
        if (face := scene.faces.get(track.track_id)) is not None:
            _draw_face(frame, face, track.box.width, color, thickness)
            lines.append(describe(face))
        for index, line in enumerate(reversed(lines)):
            cv2.putText(
                frame,
                line,
                (x, max(y - round((8 + 22 * index) * scale), round(20 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale,
                color,
                thickness,
            )
        if caption is not None and caption[0] == track.track_id:
            _draw_speech(frame, caption[1], (x, bottom), color, scale, thickness)
    overlay = (
        f"{timings.fps:.0f} fps  capture {timings.capture_ms:.1f} ms  "
        f"detect {timings.detect_ms:.1f} ms  embed {timings.embed_ms:.1f} ms  "
        f"faces {timings.faces_ms:.1f} ms  bodies {timings.bodies_ms:.1f} ms"
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


def describe(face: FaceFeatures) -> str:
    """Eye and mouth openness, gaze and expression, in a line."""
    return (
        f"eyes {face.left_eye_open:.2f} {face.right_eye_open:.2f}  mouth {face.mouth_open:.2f}  "
        f"gaze {face.gaze.yaw:+.0f} {face.gaze.pitch:+.0f}  "
        f"{face.expression} {face.expression_score:.2f}"
    )


def label(track: Track) -> str:
    if track.match is None:
        return f"#{track.track_id} unknown"
    person = track.match.person
    return f"#{track.track_id} {person.name} ({str(person.user_id)[:8]}) {track.match.score:.2f}"


def _color(track: Track) -> tuple[int, int, int]:
    if not track.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if track.match is None else _KNOWN


def _draw_speech(
    frame: Image,
    text: str,
    under: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    """Draw a sentence in a dark box hanging under a face box, wrapped to a few lines."""
    font_scale = 0.6 * scale
    lines = textwrap.wrap(f'"{text}"', _SPEECH_LINE_CHARS)
    (_, line_height), baseline = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        for line in lines
    ]
    pad, step = round(6 * scale), line_height + baseline + round(4 * scale)
    height, width = frame.shape[:2]
    # Keep the box inside the image, under the face if there is room, else above its bottom edge.
    box_width, box_height = max(widths) + 2 * pad, step * len(lines) + pad
    left = min(max(under[0], 0), max(width - box_width, 0))
    top = min(under[1] + round(4 * scale), max(height - box_height, 0))
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), color, thickness)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (left + pad, top + pad + line_height + index * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def _draw_body(frame: Image, body: Body, color: tuple[int, int, int], thickness: int) -> None:
    for keypoints, connections in (
        (body.keypoints, BODY_CONNECTIONS),
        (body.left_hand.keypoints, HAND_CONNECTIONS),
        (body.right_hand.keypoints, HAND_CONNECTIONS),
    ):
        for start, end in connections:
            a, b = keypoints[start], keypoints[end]
            if a is not None and b is not None:
                cv2.line(frame, _point(a.x, a.y), _point(b.x, b.y), color, thickness)
        for keypoint in keypoints:
            if keypoint is not None:
                cv2.circle(frame, _point(keypoint.x, keypoint.y), thickness + 1, color, -1)


def _draw_face(
    frame: Image, face: FaceFeatures, size: float, color: tuple[int, int, int], thickness: int
) -> None:
    """Outline the eyes and lips, mark the irises, and draw the gaze from between the eyes."""
    points = np.array(face.landmarks, np.float32)
    for contour in _FACE_CONTOURS:
        cv2.polylines(frame, [points[list(contour)].round().astype(np.int32)], True, color, 1)
    for iris in _IRISES:
        cv2.circle(frame, _point(*points[iris]), thickness, (255, 255, 255), -1)
    origin = points[list(_IRISES)].mean(axis=0)
    yaw, pitch = np.radians(face.gaze.yaw), np.radians(face.gaze.pitch)
    tip = origin + size * np.array([np.sin(yaw), -np.sin(pitch)])
    cv2.arrowedLine(frame, _point(*origin), _point(*tip), (255, 255, 0), thickness, tipLength=0.2)


def _point(x: float, y: float) -> tuple[int, int]:
    return round(x), round(y)
