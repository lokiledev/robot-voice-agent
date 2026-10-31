from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Final, Self
from uuid import UUID

import httpx
from pydantic import AnyHttpUrl
from websockets.asyncio.client import ClientConnection, connect

from rva_pydantic import RVABaseModel
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import Task, TaskEvent, TaskRequest


class HttpVlaClientConfig(RVABaseModel):
    base_url: AnyHttpUrl
    """Root of the VLA API, e.g. http://127.0.0.1:8800."""


class HttpVlaClient:
    """`VlaClient` over the VLA's HTTP API, with task events over a WebSocket."""

    def __init__(self, config: HttpVlaClientConfig) -> None:
        base_url = httpx.URL(str(config.base_url))
        self._http: Final = httpx.AsyncClient(base_url=base_url)
        self._events_url: Final = base_url.copy_with(
            scheme="wss" if base_url.scheme == "https" else "ws", path="/events"
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

    async def start(self, request: TaskRequest) -> Task:
        response = await self._http.post(
            "/tasks",
            content=request.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        if response.status_code == httpx.codes.CONFLICT:
            raise VlaBusyError(Task.model_validate_json(response.content).id)
        response.raise_for_status()
        return Task.model_validate_json(response.content)

    async def cancel(self, task_id: UUID) -> Task:
        return self._task_from(task_id, await self._http.post(f"/tasks/{task_id}/cancel"))

    async def get(self, task_id: UUID) -> Task:
        return self._task_from(task_id, await self._http.get(f"/tasks/{task_id}"))

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[TaskEvent]]:
        # The server subscribes before completing the handshake, so no event is missed once
        # `connect` returns.
        async with connect(str(self._events_url)) as websocket:
            yield _events(websocket)

    @staticmethod
    def _task_from(task_id: UUID, response: httpx.Response) -> Task:
        if response.status_code == httpx.codes.NOT_FOUND:
            raise UnknownTaskError(task_id)
        response.raise_for_status()
        return Task.model_validate_json(response.content)


async def _events(websocket: ClientConnection) -> AsyncIterator[TaskEvent]:
    async for message in websocket:
        yield TaskEvent.model_validate_json(message)
