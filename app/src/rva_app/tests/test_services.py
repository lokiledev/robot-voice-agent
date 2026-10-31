import asyncio
import socket
import sys
from pathlib import Path

import pytest

from rva_app.servers import Server, is_ready
from rva_app.services import Service, ServiceFailedError, services_running


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
            url=f"http://127.0.0.1:{port}/",
            how_to_start="run it",
            ready_when="answers",
        ),
        command=lambda: command,
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
