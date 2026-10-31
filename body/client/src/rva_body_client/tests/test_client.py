import asyncio
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import AnyHttpUrl

from rva_body_client.client import HttpBodyClient, HttpBodyClientConfig
from rva_body_schemas.abc.client import BodyClient
from rva_body_schemas.errors import BodyBusyError
from rva_body_schemas.motion import BodyState, GoToLocation, MotionRequest, PlayAnimation
from rva_vla_mock.body import BodyMock
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_mock.server import build_app
from rva_vla_schemas.task import TaskRequest, TaskRunning, TaskSucceeded


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncGenerator[AnyHttpUrl]:
    # Listening before uvicorn starts queues early connections, so no need to wait for startup.
    sock = socket.create_server(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        yield AnyHttpUrl(f"http://127.0.0.1:{sock.getsockname()[1]}")
    finally:
        server.should_exit = True
        await serving


async def test_motion_life_gaze_and_busy_body_over_http() -> None:
    commands = asyncio.Queue[str]()
    vla = TaskRunner(TerminalPolicy(commands, fallback=None))
    body_mock = BodyMock(
        vla, TerminalPolicy(commands, fallback=None), TerminalPolicy(commands, fallback=None)
    )

    async with (
        _serve(build_app(vla, body_mock)) as url,
        HttpBodyClient(HttpBodyClientConfig(base_url=url)) as client,
    ):
        body: BodyClient = client
        await body.look_at(UUID(int=7))
        async with body.events() as events:
            motion = await body.start(MotionRequest(action=GoToLocation(location="shelf")))
            with pytest.raises(BodyBusyError) as busy:
                await body.start(MotionRequest(action=PlayAnimation(animation="dance")))
            # Walking stops looking at someone.
            assert await body.state() == BodyState(motion=motion, looking_at=None)
            commands.put_nowait("s")
            statuses = [(await anext(events)).status for _ in range(2)]

        await vla.start(TaskRequest(instruction="pick up the red cup"))
        with pytest.raises(BodyBusyError) as vla_busy:
            await body.start(MotionRequest(action=PlayAnimation(animation="dance")))

    assert busy.value.busy_with == "go to the shelf"
    assert statuses == [TaskRunning(), TaskSucceeded()]
    assert vla_busy.value.busy_with == "pick up the red cup"
