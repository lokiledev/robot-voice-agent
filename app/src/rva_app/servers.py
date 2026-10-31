import asyncio
from typing import Literal, assert_never

import httpx

from rva_pydantic import RVABaseModel


class Server(RVABaseModel):
    name: str
    url: str
    """Address to probe."""
    how_to_start: str
    ready_when: Literal["answers", "route_exists", "succeeds"]
    """Any HTTP answer, even an error; anything but a 404, for a route of an API; a 2xx status,
    for a health check that fails while the server is still loading."""


class ServersUnavailableError(Exception):
    def __init__(self, servers: tuple[Server, ...]) -> None:
        lines = [f"  {server.name} at {server.url}: {server.how_to_start}" for server in servers]
        hint = "Start them all with `uv run rva up`, or each in a terminal of its own."
        super().__init__("\n".join(["Not running:", *lines, hint]))
        self.servers = servers


async def require_servers(*servers: Server) -> None:
    """Check that each server answers HTTP, so a missing one fails before any model loads.

    Raises:
        ServersUnavailableError: Listing every server that did not answer.
    """
    answered = await asyncio.gather(*(is_ready(server) for server in servers))
    missing = tuple(server for server, ok in zip(servers, answered, strict=True) if not ok)
    if missing:
        raise ServersUnavailableError(missing)


async def is_ready(server: Server) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(server.url)
    except httpx.TransportError:
        return False
    match server.ready_when:
        case "answers":
            return True
        case "route_exists":
            return response.status_code != httpx.codes.NOT_FOUND
        case "succeeds":
            return response.is_success
        case _:
            assert_never(server.ready_when)
