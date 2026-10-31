from datetime import UTC, datetime
from typing import Final, assert_never, override

import rerun as rr
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rva_schemas.record import DialogueLine, Measurement, Record, TaskUpdate, ToolInvocation

_TIMELINE: Final = "wall"


class RerunRecorder:
    """`Recorder` writing each entry to rerun, on the wall-clock timeline."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording

    def record(self, entry: Record) -> None:
        self._recording.set_time(_TIMELINE, timestamp=datetime.now(UTC))
        match entry:
            case DialogueLine():
                self._recording.log(f"dialogue/{entry.speaker}", rr.TextLog(entry.text))
            case ToolInvocation():
                text = f"{entry.name}({entry.arguments}) -> {entry.result}"
                self._recording.log("agent/tools", rr.TextLog(text))
            case TaskUpdate():
                level = "ERROR" if entry.task.status.state == "failed" else "INFO"
                text = f"{entry.task.instruction!r}: {entry.task.status.model_dump_json()}"
                self._recording.log("vla/task", rr.TextLog(text, level=level))
            case Measurement():
                self._recording.log(f"measurements/{entry.name}", rr.Scalars(entry.value))
            case _:
                assert_never(entry)


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
