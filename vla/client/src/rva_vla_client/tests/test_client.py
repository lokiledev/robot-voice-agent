import asyncio
from uuid import UUID

import pytest
from fastapi import FastAPI

from rva_vla_client.client import HttpVlaClient, HttpVlaClientConfig
from rva_vla_client.tests.serve import serve_locally
from rva_vla_mock.body import BodyMock
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_mock.server import build_app
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import TaskCancelled, TaskFailed, TaskRequest, TaskRunning

PICK_CUP = TaskRequest(instruction="pick up the red cup")


def _app(runner: TaskRunner) -> FastAPI:
    idle = TerminalPolicy(asyncio.Queue(), fallback=None)
    return build_app(runner, BodyMock(runner, idle, idle))


async def test_task_life_over_http_and_websocket() -> None:
    commands = asyncio.Queue[str]()

    async with (
        serve_locally(_app(TaskRunner(TerminalPolicy(commands, fallback=None)))) as url,
        HttpVlaClient(HttpVlaClientConfig(base_url=url)) as client,
    ):
        vla: VlaClient = client
        async with vla.events() as events:
            task = await vla.start(PICK_CUP)
            with pytest.raises(VlaBusyError) as busy:
                await vla.start(TaskRequest(instruction="open the drawer"))
            commands.put_nowait("f the grasp slipped")

            statuses = [(await anext(events)).status for _ in range(2)]

        assert busy.value.running_task_id == task.id
        assert statuses == [TaskRunning(), TaskFailed(cause="error", reason="the grasp slipped")]
        assert (await vla.get(task.id)).status == TaskFailed(
            cause="error", reason="the grasp slipped"
        )


async def test_cancel_over_http() -> None:
    async with (
        serve_locally(_app(TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)))) as url,
        HttpVlaClient(HttpVlaClientConfig(base_url=url)) as client,
    ):
        task = await client.start(PICK_CUP)

        assert (await client.cancel(task.id)).status == TaskCancelled()


async def test_unknown_task_over_http() -> None:
    task_id = UUID("6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11")

    async with (
        serve_locally(_app(TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)))) as url,
        HttpVlaClient(HttpVlaClientConfig(base_url=url)) as client,
    ):
        with pytest.raises(UnknownTaskError):
            await client.cancel(task_id)
