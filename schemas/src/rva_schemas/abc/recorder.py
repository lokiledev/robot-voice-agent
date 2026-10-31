from typing import Protocol

from rva_schemas.record import Record


class Recorder(Protocol):
    """Session recording. Timings go through OpenTelemetry spans instead.

    May be called from any thread, e.g. the camera's.
    """

    def record(self, entry: Record) -> None: ...
