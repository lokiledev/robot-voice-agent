import asyncio

import httpx

from rva_pydantic import RVABaseModel


class Server(RVABaseModel):
    name: str
    url: str
    """Address to probe."""
    how_to_start: str
    route_must_exist: bool
    """Whether a 404 counts as missing: for a route of the API rather than its root."""


class ServersUnavailableError(Exception):
    def __init__(self, servers: tuple[Server, ...]) -> None:
        lines = [f"  {server.name} at {server.url}: {server.how_to_start}" for server in servers]
        super().__init__("\n".join(["Not running:", *lines, "Each server needs its own terminal."]))
        self.servers = servers


async def require_servers(*servers: Server) -> None:
    """Check that each server answers HTTP, so a missing one fails before any model loads.

    Raises:
        ServersUnavailableError: Listing every server that did not answer.
    """
    answered = await asyncio.gather(*(_answers(server) for server in servers))
    missing = tuple(server for server, ok in zip(servers, answered, strict=True) if not ok)
    if missing:
        raise ServersUnavailableError(missing)


async def _answers(server: Server) -> bool:
    # Otherwise any HTTP response, even an error status, means something is listening.
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(server.url)
    except httpx.TransportError:
        return False
    return not (server.route_must_exist and response.status_code == httpx.codes.NOT_FOUND)
