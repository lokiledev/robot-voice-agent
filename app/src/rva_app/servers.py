import asyncio
from typing import Annotated, Literal, assert_never

import httpx
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_client.client import robot_is_up


class HttpProbe(RVABaseModel):
    kind: Literal["http"] = "http"
    url: str
    """Address to probe."""
    ready_when: Literal["answers", "route_exists", "succeeds"]
    """Any HTTP answer, even an error; anything but a 404, for a route of an API; a 2xx status,
    for a health check that fails while the server is still loading."""


class RobotProbe(RVABaseModel):
    """A robot serving its kickmsg topics."""

    kind: Literal["robot"] = "robot"
    namespace: str


class Server(RVABaseModel):
    name: str
    probe: Annotated[HttpProbe | RobotProbe, Field(discriminator="kind")]
    how_to_start: str


class ServersUnavailableError(Exception):
    def __init__(self, servers: tuple[Server, ...]) -> None:
        lines = [f"  {server.name} {_where(server)}: {server.how_to_start}" for server in servers]
        hint = "Start them all with `uv run lune`, or each in a terminal of its own."
        super().__init__("\n".join(["Not running:", *lines, hint]))
        self.servers = servers


async def require_servers(*servers: Server) -> None:
    """Check that each server is up, so a missing one fails before any model loads.

    Raises:
        ServersUnavailableError: Listing every server that did not answer.
    """
    answered = await asyncio.gather(*(is_ready(server) for server in servers))
    missing = tuple(server for server, ok in zip(servers, answered, strict=True) if not ok)
    if missing:
        raise ServersUnavailableError(missing)


async def is_ready(server: Server) -> bool:
    match server.probe:
        case HttpProbe():
            return await _answers(server.probe)
        case RobotProbe():
            return await asyncio.to_thread(robot_is_up, server.probe.namespace)
        case _:
            assert_never(server.probe)


async def _answers(probe: HttpProbe) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(probe.url)
    except httpx.TransportError:
        return False
    match probe.ready_when:
        case "answers":
            return True
        case "route_exists":
            return response.status_code != httpx.codes.NOT_FOUND
        case "succeeds":
            return response.is_success
        case _:
            assert_never(probe.ready_when)


def _where(server: Server) -> str:
    match server.probe:
        case HttpProbe():
            return f"at {server.probe.url}"
        case RobotProbe():
            return f"in kickmsg namespace {server.probe.namespace!r}"
        case _:
            assert_never(server.probe)
