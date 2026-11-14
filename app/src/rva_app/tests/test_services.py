import asyncio
import socket
import sys
from pathlib import Path

import pytest

from rva_app.servers import HttpProbe, Server, is_ready
from rva_app.services import Service, ServiceFailedError, run_last, services_running


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port


def _web_server(port: int, command: list[str]) -> Service:
    return Service(
        server=Server(
            name="Web server",
            probe=HttpProbe(url=f"http://127.0.0.1:{port}/", ready_when="answers"),
            how_to_start="run it",
        ),
        command=lambda: command,
        notifies=False,
        environment={},
        directory=None,
        startup_s=10,
    )


def _never_started() -> list[str]:
    raise AssertionError("A service already running must not be started")


async def _answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
    await writer.drain()
    writer.close()


async def test_starts_a_missing_service_waits_until_ready_and_stops_it_after(
    tmp_path: Path,
) -> None:
    port = _free_port()
    service = _web_server(
        port, [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    )

    async with services_running([service], tmp_path):
        ready_inside = await is_ready(service.server)

    assert ready_inside
    assert not await is_ready(service.server)
    assert (tmp_path / "web-server.log").exists()


async def test_leaves_a_service_that_already_runs_alone(tmp_path: Path) -> None:
    server = await asyncio.start_server(_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = _web_server(port, []).model_copy(update={"command": _never_started})

    async with server:
        async with services_running([service], tmp_path):
            pass

        assert await is_ready(service.server)


async def test_a_service_that_exits_fails_and_points_to_its_output(tmp_path: Path) -> None:
    service = _web_server(
        _free_port(), [sys.executable, "-c", "print('model not found'); raise SystemExit(3)"]
    )

    with pytest.raises(ServiceFailedError, match="Web server exited with code 3"):
        async with services_running([service], tmp_path):
            pass

    assert (tmp_path / "web-server.log").read_text() == "model not found\n"


def _notifying(code: str) -> list[str]:
    """A process of this project: runs `code`, in which `notify_ready()` tells it is ready."""
    return [sys.executable, "-c", f"from rva_app.ready import notify_ready\n{code}"]


def _notifying_service(command: list[str]) -> Service:
    # Probed only to tell whether it runs already: nothing ever answers there.
    return _web_server(_free_port(), command).model_copy(update={"notifies": True})


async def test_a_service_that_notifies_is_ready_when_it_says_so(tmp_path: Path) -> None:
    marker = tmp_path / "before-ready"
    service = _notifying_service(
        _notifying(
            f"import pathlib, time\ntime.sleep(0.3)\npathlib.Path({str(marker)!r}).touch()\n"
            "notify_ready()\ntime.sleep(60)"
        )
    )

    async with services_running([service], tmp_path):
        ready_inside = marker.exists()

    assert ready_inside


async def test_a_service_that_exits_without_notifying_fails(tmp_path: Path) -> None:
    service = _notifying_service(_notifying("print('no camera')"))

    with pytest.raises(ServiceFailedError, match="Web server exited with code 0"):
        async with services_running([service], tmp_path):
            pass


async def test_the_last_process_runs_until_it_exits_after_saying_it_is_ready() -> None:
    told: list[str] = []

    code = await run_last(
        "The agent",
        _notifying("notify_ready()\nraise SystemExit(7)"),
        startup_s=10,
        when_ready=lambda: told.append("ready"),
    )

    assert code == 7
    assert told == ["ready"]


async def test_the_last_process_failing_before_it_is_ready_is_an_error() -> None:
    told: list[str] = []

    with pytest.raises(ServiceFailedError, match="The agent exited with code 3"):
        await run_last(
            "The agent",
            _notifying("raise SystemExit(3)"),
            startup_s=10,
            when_ready=lambda: told.append("ready"),
        )

    assert told == []
