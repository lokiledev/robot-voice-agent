import asyncio
import socket
from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import pytest

from rva_app.servers import (
    HttpProbe,
    RobotProbe,
    Server,
    ServersUnavailableError,
    is_ready,
    require_servers,
)
from rva_robot_sim.bus import RobotBus


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port


@pytest.fixture
def namespace() -> Iterator[str]:
    """A kickmsg namespace of its own for each test, removed after."""
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


async def _answer_404(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
    await writer.drain()
    writer.close()


async def test_lists_every_missing_server_with_how_to_start_it() -> None:
    llm_port, vla_port = _closed_port(), _closed_port()
    llm = Server(
        name="LLM server",
        probe=HttpProbe(url=f"http://127.0.0.1:{llm_port}/v1", ready_when="answers"),
        how_to_start="run A",
    )
    vla = Server(
        name="VLA",
        probe=HttpProbe(url=f"http://127.0.0.1:{vla_port}", ready_when="answers"),
        how_to_start="run B",
    )
    robot = Server(name="Robot", probe=RobotProbe(namespace="nobody_here"), how_to_start="run C")

    with pytest.raises(ServersUnavailableError) as error:
        await require_servers(llm, vla, robot)

    assert str(error.value) == (
        "Not running:\n"
        f"  LLM server at http://127.0.0.1:{llm_port}/v1: run A\n"
        f"  VLA at http://127.0.0.1:{vla_port}: run B\n"
        "  Robot in kickmsg namespace 'nobody_here': run C\n"
        "Start them all with `uv run lune`, or each in a terminal of its own."
    )


async def test_any_http_answer_counts_as_running() -> None:
    server = await asyncio.start_server(_answer_404, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async with server:
        await require_servers(
            Server(
                name="VLA",
                probe=HttpProbe(url=f"http://127.0.0.1:{port}", ready_when="answers"),
                how_to_start="run B",
            )
        )


async def test_a_missing_route_counts_as_missing_when_it_must_exist() -> None:
    server = await asyncio.start_server(_answer_404, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/body/state"

    async with server:
        with pytest.raises(ServersUnavailableError):
            await require_servers(
                Server(
                    name="Body API",
                    probe=HttpProbe(url=url, ready_when="route_exists"),
                    how_to_start="restart",
                )
            )


async def _answer_503(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
    await writer.drain()
    writer.close()


async def test_a_loading_server_answers_but_is_not_ready_to_succeed() -> None:
    server = await asyncio.start_server(_answer_503, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/health"

    async with server:
        answers = await is_ready(
            Server(name="LLM", probe=HttpProbe(url=url, ready_when="answers"), how_to_start="")
        )
        succeeds = await is_ready(
            Server(name="LLM", probe=HttpProbe(url=url, ready_when="succeeds"), how_to_start="")
        )

    assert (answers, succeeds) == (True, False)


async def test_a_robot_is_ready_while_it_serves_its_topics(namespace: str) -> None:
    server = Server(name="Robot", probe=RobotProbe(namespace=namespace), how_to_start="")

    before = await is_ready(server)
    bus = RobotBus(namespace, (4, 2))
    during = await is_ready(server)
    bus.close()
    after = await is_ready(server)

    assert (before, during, after) == (False, True, False)
