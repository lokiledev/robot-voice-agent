import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import rerun as rr
from loguru import logger
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from rva_observability.rerun_sink import RerunRecorder, RerunSpanProcessor
from rva_pydantic import RVABaseModel

_APPLICATION_ID: Final = "robot-voice-agent"
_RECORDING: Final = "session.rrd"


class ObservabilityConfig(RVABaseModel):
    captures_dir: Path
    """Where session directories are created; `~` is expanded."""


def start_session(
    config: ObservabilityConfig, name: str, viewer: bool, console_level: str
) -> RerunRecorder:
    """Create a session directory and send logs, spans and records into it.

    The directory holds `session.rrd` and `agent.log`. With `viewer`, the recording also streams to
    a rerun viewer, opened if none is.
    """
    directory = _session_directory(config, name)

    logger.remove()
    logger.add(sys.stderr, level=console_level)
    logger.add(directory / "agent.log", level="DEBUG")

    recording = _recording(directory, viewer)

    provider = TracerProvider()
    provider.add_span_processor(RerunSpanProcessor(recording))
    trace.set_tracer_provider(provider)

    logger.info("Recording session to {}", directory)
    return RerunRecorder(recording)


def record_session(config: ObservabilityConfig, name: str, viewer: bool) -> rr.RecordingStream:
    """A rerun recording of its own, for a process other than the agent's to log its telemetry.

    Written to `session.rrd` in a session directory, as the agent's is; with `viewer`, streamed to
    a rerun viewer too, opened if none is.
    """
    directory = _session_directory(config, name)
    logger.info("Recording {} to {}", name, directory)
    return _recording(directory, viewer)


def _session_directory(config: ObservabilityConfig, name: str) -> Path:
    directory = config.captures_dir.expanduser() / f"{datetime.now():%Y-%m-%dT%H%M%S}-{name}"
    directory.mkdir(parents=True)
    return directory


def _recording(directory: Path, viewer: bool) -> rr.RecordingStream:
    recording = rr.RecordingStream(_APPLICATION_ID)
    sinks: list[rr.FileSink | rr.GrpcSink] = [rr.FileSink(directory / _RECORDING)]
    if viewer:
        open_viewer()
        sinks.append(rr.GrpcSink())
    recording.set_sinks(*sinks)
    return recording


def open_viewer() -> None:
    """Open a rerun viewer, unless one is open already, and return once it takes recordings.

    It outlives this process, to look at what was recorded.
    """
    # Through a recording: the global rr.spawn ignores the call when no global recording was
    # initialized, which this app never does.
    rr.RecordingStream(_APPLICATION_ID).spawn(connect=False)


def latest_recording(config: ObservabilityConfig) -> Path | None:
    """The recording of the session started last, or `None` before any session."""
    recordings = list(config.captures_dir.expanduser().glob(f"*/{_RECORDING}"))
    if not recordings:
        return None
    return max(recordings, key=lambda recording: recording.stat().st_mtime)
