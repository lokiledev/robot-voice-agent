import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import Task, TaskEvent, TaskRequest


def build_app(runner: TaskRunner) -> FastAPI:
    """HTTP and WebSocket API of the VLA, served from `runner`.

    `POST /tasks` answers 409 with the running task when the VLA is busy, and task routes answer
    404 for an unknown id.
    """
    app = FastAPI(title="Mock VLA")

    @app.post("/tasks", status_code=201, response_model=Task, responses={409: {"model": Task}})
    async def start(request: TaskRequest) -> Task | JSONResponse:
        try:
            return await runner.start(request)
        except VlaBusyError as error:
            running = await runner.get(error.running_task_id)
            return JSONResponse(running.model_dump(mode="json"), status_code=409)

    @app.get("/tasks/{task_id}")
    async def get(task_id: UUID) -> Task:
        try:
            return await runner.get(task_id)
        except UnknownTaskError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/tasks/{task_id}/cancel")
    async def cancel(task_id: UUID) -> Task:
        try:
            return await runner.cancel(task_id)
        except UnknownTaskError as error:
            raise HTTPException(404, str(error)) from error

    @app.websocket("/events")
    async def events(websocket: WebSocket) -> None:
        # Subscribe before accepting, so a client that has connected cannot miss an event.
        async with runner.events() as task_events:
            await websocket.accept()
            forwarding = asyncio.create_task(_forward(task_events, websocket))
            try:
                # Clients never send anything, so the next message is the disconnect. Waiting for it
                # ends the handler even while no event is coming.
                while (await websocket.receive())["type"] != "websocket.disconnect":
                    pass
            finally:
                forwarding.cancel()

    return app


async def _forward(task_events: AsyncIterator[TaskEvent], websocket: WebSocket) -> None:
    async for event in task_events:
        await websocket.send_text(event.model_dump_json())
