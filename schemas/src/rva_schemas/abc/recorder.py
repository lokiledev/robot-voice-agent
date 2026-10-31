from typing import Protocol

from rva_schemas.record import Record


class Recorder(Protocol):
    """Session recording. Timings go through OpenTelemetry spans instead."""

    def record(self, entry: Record) -> None: ...
