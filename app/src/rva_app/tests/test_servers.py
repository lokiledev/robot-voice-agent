import asyncio
import socket

import pytest

from rva_app.servers import Server, ServersUnavailableError, require_servers


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert isinstance(port, int)
    return port


async def _answer_404(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
    await writer.drain()
    writer.close()


async def test_lists_every_missing_server_with_how_to_start_it() -> None:
    llm_port, vla_port = _closed_port(), _closed_port()
    llm = Server(name="LLM server", url=f"http://127.0.0.1:{llm_port}/v1", how_to_start="run A")
    vla = Server(name="VLA", url=f"http://127.0.0.1:{vla_port}", how_to_start="run B")

    with pytest.raises(ServersUnavailableError) as error:
        await require_servers(llm, vla)

    assert str(error.value) == (
        "Not running:\n"
        f"  LLM server at http://127.0.0.1:{llm_port}/v1: run A\n"
        f"  VLA at http://127.0.0.1:{vla_port}: run B\n"
        "Each server needs its own terminal."
    )


async def test_any_http_answer_counts_as_running() -> None:
    server = await asyncio.start_server(_answer_404, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async with server:
        await require_servers(
            Server(name="VLA", url=f"http://127.0.0.1:{port}", how_to_start="run B")
        )
