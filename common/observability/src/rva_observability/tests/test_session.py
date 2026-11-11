import os
from pathlib import Path

from rva_observability.session import ObservabilityConfig, latest_recording


def test_the_latest_recording_is_of_the_session_started_last(tmp_path: Path) -> None:
    for age, name in enumerate(("2026-11-11T151200-voice", "2026-11-11T150416-see")):
        (tmp_path / name).mkdir()
        recording = tmp_path / name / "session.rrd"
        recording.touch()
        os.utime(recording, (1_000_000 - age, 1_000_000 - age))
    # The servers' logs sit beside the sessions, without a recording.
    (tmp_path / "servers").mkdir()

    assert latest_recording(ObservabilityConfig(captures_dir=tmp_path)) == (
        tmp_path / "2026-11-11T151200-voice" / "session.rrd"
    )


def test_there_is_no_latest_recording_before_any_session(tmp_path: Path) -> None:
    assert latest_recording(ObservabilityConfig(captures_dir=tmp_path)) is None
