import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from rva_body_schemas.errors import BodyBusyError, UnknownMotionError
from rva_body_schemas.motion import BodyState, GazeRequest, Motion, MotionRequest
from rva_pydantic import RVABaseModel
from rva_vla_mock.body import BodyMock
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import Task, TaskRequest


def build_app(runner: TaskRunner, body: BodyMock) -> FastAPI:
    """HTTP and WebSocket APIs of the VLA, served from `runner`, and of the body, from `body`.

    `POST /tasks` answers 409 with the running task when the VLA is busy, `POST /body/motions`
    answers 409 with what the body is busy with, and routes with an id answer 404 for an unknown
    one.
    """
    app = FastAPI(title="Mock robot")

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
            await _stream(task_events, websocket)

    @app.post("/body/motions", status_code=201, response_model=Motion, responses={409: {}})
    async def start_motion(request: MotionRequest) -> Motion | JSONResponse:
        try:
            return await body.start(request)
        except BodyBusyError as error:
            return JSONResponse({"busy_with": error.busy_with}, status_code=409)

    @app.post("/body/motions/{motion_id}/cancel")
    async def cancel_motion(motion_id: UUID) -> Motion:
        try:
            return await body.cancel(motion_id)
        except UnknownMotionError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/body/stay", status_code=204)
    async def stay() -> None:
        await body.stay()

    @app.put("/body/gaze", status_code=204)
    async def look_at(request: GazeRequest) -> None:
        await body.look_at(request.user_id)

    @app.delete("/body/gaze", status_code=204)
    async def stop_looking() -> None:
        await body.stop_looking()

    @app.get("/body/state")
    async def state() -> BodyState:
        return await body.state()

    @app.websocket("/body/events")
    async def motion_events(websocket: WebSocket) -> None:
        async with body.events() as events:
            await _stream(events, websocket)

    return app


async def _stream(events: AsyncIterator[RVABaseModel], websocket: WebSocket) -> None:
    """Send each event as JSON until the client disconnects."""
    await websocket.accept()
    forwarding = asyncio.create_task(_forward(events, websocket))
    try:
        # Clients never send anything, so the next message is the disconnect. Waiting for it
        # ends the handler even while no event is coming.
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass
    finally:
        forwarding.cancel()


async def _forward(events: AsyncIterator[RVABaseModel], websocket: WebSocket) -> None:
    async for event in events:
        await websocket.send_text(event.model_dump_json())
