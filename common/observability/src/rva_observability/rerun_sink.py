from datetime import UTC, datetime
from typing import Final, assert_never, override

import rerun as rr
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rva_body_schemas.motion import describe
from rva_schemas.record import (
    AgentStarted,
    CameraFrame,
    FaceBox,
    FaceBoxes,
    FaceUpdate,
    GazeUpdate,
    Measurement,
    MotionUpdate,
    MuteUpdate,
    Record,
    RobotLine,
    RobotSentence,
    Speaker,
    TaskUpdate,
    ToolInvocation,
    UserLine,
)

_TIMELINE: Final = "wall"
_CAMERA: Final = "camera"
"""Image and face boxes share this entity, so rerun draws them in one 2D view."""
# RGB, as in the debug window.
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)


class RerunRecorder:
    """`Recorder` writing each entry to rerun, on the wall-clock timeline."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording

    def record(self, entry: Record) -> None:
        self._recording.set_time(_TIMELINE, timestamp=datetime.now(UTC))
        match entry:
            case AgentStarted():
                tools = ", ".join(tool.name for tool in entry.tools)
                self._recording.log("agent/started", rr.TextLog(f"{entry.name}, tools: {tools}"))
            case UserLine():
                text = f"[{_who(entry.speaker)}] {entry.text}"
                self._recording.log("dialogue/user", rr.TextLog(text))
            case RobotSentence():
                self._recording.log("dialogue/robot/sentences", rr.TextLog(entry.text))
            case RobotLine():
                text = f"{entry.text} [interrupted]" if entry.interrupted else entry.text
                self._recording.log("dialogue/robot", rr.TextLog(text))
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
            case GazeUpdate():
                person = entry.looking_at
                text = (
                    "stopped looking"
                    if person is None
                    else f"looking at {person.name} ({person.user_id})"
                )
                self._recording.log("body/gaze", rr.TextLog(text))
            case MuteUpdate():
                self._recording.log("agent/muted", rr.TextLog("muted" if entry.muted else "talks"))
            case Measurement():
                self._recording.log(f"measurements/{entry.name}", rr.Scalars(entry.value))
            case FaceUpdate():
                self._recording.log("vision/faces", rr.TextLog(entry.event.model_dump_json()))
            case FaceBoxes():
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


def _who(speaker: Speaker | None) -> str:
    if speaker is None:
        return "nobody in view"
    if speaker.person is None:
        return f"unknown, track {speaker.track_id}"
    role = ", operator" if speaker.operator else ""
    return f"{speaker.person.name}, track {speaker.track_id}{role}"


def _label(box: FaceBox) -> str:
    who = "unknown" if box.person is None else f"{box.person.name} ({str(box.person.user_id)[:8]})"
    said = "" if box.said is None else f': "{box.said}"'
    return f"#{box.track_id} {who}{said}"


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
