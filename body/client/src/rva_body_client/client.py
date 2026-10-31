from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Final, Self
from uuid import UUID

import httpx
from pydantic import AnyHttpUrl
from websockets.asyncio.client import ClientConnection, connect

from rva_body_schemas.errors import BodyBusyError, UnknownMotionError
from rva_body_schemas.motion import BodyState, GazeRequest, Motion, MotionEvent, MotionRequest
from rva_pydantic import RVABaseModel


class HttpBodyClientConfig(RVABaseModel):
    base_url: AnyHttpUrl
    """Root of the robot API, e.g. http://127.0.0.1:8800; the body routes are under /body."""


class _Busy(RVABaseModel):
    busy_with: str


class HttpBodyClient:
    """`BodyClient` over the robot's HTTP API, with motion events over a WebSocket."""

    def __init__(self, config: HttpBodyClientConfig) -> None:
        base_url = httpx.URL(str(config.base_url))
        self._http: Final = httpx.AsyncClient(base_url=base_url)
        self._events_url: Final = base_url.copy_with(
            scheme="wss" if base_url.scheme == "https" else "ws", path="/body/events"
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def start(self, request: MotionRequest) -> Motion:
        response = await self._http.post(
            "/body/motions",
            content=request.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        if response.status_code == httpx.codes.CONFLICT:
            raise BodyBusyError(_Busy.model_validate_json(response.content).busy_with)
        response.raise_for_status()
        return Motion.model_validate_json(response.content)

    async def cancel(self, motion_id: UUID) -> Motion:
        response = await self._http.post(f"/body/motions/{motion_id}/cancel")
        if response.status_code == httpx.codes.NOT_FOUND:
            raise UnknownMotionError(motion_id)
        response.raise_for_status()
        return Motion.model_validate_json(response.content)

    async def stay(self) -> None:
        (await self._http.post("/body/stay")).raise_for_status()

    async def look_at(self, user_id: UUID) -> None:
        response = await self._http.put(
            "/body/gaze",
            content=GazeRequest(user_id=user_id).model_dump_json(),
            headers={"content-type": "application/json"},
        )
        response.raise_for_status()

    async def stop_looking(self) -> None:
        (await self._http.delete("/body/gaze")).raise_for_status()

    async def state(self) -> BodyState:
        response = await self._http.get("/body/state")
        response.raise_for_status()
        return BodyState.model_validate_json(response.content)

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[MotionEvent]]:
        # The server subscribes before completing the handshake, so no event is missed once
        # `connect` returns.
        async with connect(str(self._events_url)) as websocket:
            yield _events(websocket)


async def _events(websocket: ClientConnection) -> AsyncIterator[MotionEvent]:
    async for message in websocket:
        yield MotionEvent.model_validate_json(message)
