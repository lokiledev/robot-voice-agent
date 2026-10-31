from datetime import UTC, datetime
from typing import Final, assert_never, override

import rerun as rr
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rva_body_schemas.motion import describe
from rva_schemas.record import (
    CameraFrame,
    DialogueLine,
    FaceBox,
    FaceBoxes,
    FaceUpdate,
    Measurement,
    MotionUpdate,
    Record,
    TaskUpdate,
    ToolInvocation,
)

_TIMELINE: Final = "wall"
_CAMERA: Final = "camera"
"""Image, face boxes and caption share this entity, so rerun draws them in one 2D view."""
# RGB, as in the debug window.
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)


class RerunRecorder:
    """`Recorder` writing each entry to rerun, on the wall-clock timeline."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording
        # Height of the camera image, to place the caption; known once faces were recorded.
        self._image_height: int | None = None

    def record(self, entry: Record) -> None:
        self._recording.set_time(_TIMELINE, timestamp=datetime.now(UTC))
        match entry:
            case DialogueLine():
                self._recording.log(f"dialogue/{entry.speaker}", rr.TextLog(entry.text))
                if entry.speaker == "user" and self._image_height is not None:
                    self._caption(entry.text, self._image_height)
            case ToolInvocation():
                text = f"{entry.name}({entry.arguments}) -> {entry.result}"
                self._recording.log("agent/tools", rr.TextLog(text))
            case TaskUpdate():
                level = "ERROR" if entry.task.status.state == "failed" else "INFO"
                text = f"{entry.task.instruction!r}: {entry.task.status.model_dump_json()}"
                self._recording.log("vla/task", rr.TextLog(text, level=level))
            case MotionUpdate():
                level = "ERROR" if entry.motion.status.state == "failed" else "INFO"
                text = f"{describe(entry.motion.action)}: {entry.motion.status.model_dump_json()}"
                self._recording.log("body/motion", rr.TextLog(text, level=level))
            case Measurement():
                self._recording.log(f"measurements/{entry.name}", rr.Scalars(entry.value))
            case FaceUpdate():
                self._recording.log("vision/faces", rr.TextLog(entry.event.model_dump_json()))
            case FaceBoxes():
                self._image_height = entry.image_height
                self._face_boxes(entry.boxes)
            case CameraFrame():
                self._recording.log(
                    f"{_CAMERA}/image",
                    rr.EncodedImage(contents=entry.jpeg, media_type="image/jpeg"),
                )
            case _:
                assert_never(entry)

    def _face_boxes(self, boxes: tuple[FaceBox, ...]) -> None:
        self._recording.log(
            f"{_CAMERA}/faces",
            rr.Boxes2D(
                array=[[b.box.x, b.box.y, b.box.width, b.box.height] for b in boxes],
                array_format=rr.Box2DFormat.XYWH,
                labels=[_label(box) for box in boxes],
                colors=[_color(box) for box in boxes],
                show_labels=True,
            ),
        )

    def _caption(self, text: str, image_height: int) -> None:
        # A labeled point is how rerun puts text over an image.
        self._recording.log(
            f"{_CAMERA}/caption",
            rr.Points2D(
                [[10, image_height - 20]],
                labels=[f"User: {text}"],
                colors=[(255, 255, 255)],
                radii=1,
                show_labels=True,
            ),
        )


def _label(box: FaceBox) -> str:
    if box.person is None:
        return f"#{box.track_id} unknown"
    return f"#{box.track_id} {box.person.name} ({str(box.person.user_id)[:8]})"


def _color(box: FaceBox) -> tuple[int, int, int]:
    if not box.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if box.person is None else _KNOWN


class RerunSpanProcessor(SpanProcessor):
    """Logs the duration of each finished span to `latency/<span name>`, in milliseconds.

    A span event logs the time from the start of its span to `latency/<span name>/<event name>`,
    e.g. the time to the first token.
    """

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording

    @override
    def on_end(self, span: ReadableSpan) -> None:
        if span.start_time is None or span.end_time is None:
            return
        self._recording.set_time(_TIMELINE, timestamp=span.end_time * 1e-9)
        name = span.name.replace(":", "/")
        self._recording.log(f"latency/{name}", rr.Scalars((span.end_time - span.start_time) * 1e-6))
        for event in span.events:
            elapsed_ms = (event.timestamp - span.start_time) * 1e-6
            self._recording.log(f"latency/{name}/{event.name}", rr.Scalars(elapsed_ms))
