import os

import pytest

from rva_app.ready import READY_FD, notify_ready


def test_run_by_hand_it_tells_nobody(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(READY_FD, raising=False)

    notify_ready()


def test_it_tells_the_launcher_once(monkeypatch: pytest.MonkeyPatch) -> None:
    read, write = os.pipe()
    monkeypatch.setenv(READY_FD, str(write))

    notify_ready()
    notify_ready()

    assert os.read(read, 64) == b"ready\n"
    # Its end is closed: nothing more comes.
    assert os.read(read, 64) == b""
    os.close(read)
