import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pydantic
import pytest

from rva_vla_mock.policies import RandomPolicy, RandomPolicyConfig, TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import (
    TaskCancelled,
    TaskEvent,
    TaskFailed,
    TaskRequest,
    TaskRunning,
    TaskStatus,
    TaskSucceeded,
)

PICK_CUP = TaskRequest(instruction="pick up the red cup")


async def _next_statuses(events: AsyncIterator[TaskEvent], count: int) -> list[TaskStatus]:
    return [(await anext(events)).status for _ in range(count)]


@pytest.mark.parametrize(
    ("command", "final"),
    [
        ("s\n", TaskSucceeded()),
        ("f the grasp slipped\n", TaskFailed(reason="the grasp slipped")),
        ("f\n", TaskFailed(reason="no reason given")),
    ],
)
async def test_terminal_command_ends_the_task(command: str, final: TaskStatus) -> None:
    commands = asyncio.Queue[str]()
    runner: VlaClient = TaskRunner(TerminalPolicy(commands))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        assert await _next_statuses(events, 1) == [TaskRunning()]
        commands.put_nowait("what\n")
        commands.put_nowait(command)
        assert await _next_statuses(events, 1) == [final]

    assert (await runner.get(task.id)).status == final


async def test_start_while_busy_names_the_running_task() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue()))
    task = await runner.start(PICK_CUP)

    with pytest.raises(VlaBusyError) as error:
        await runner.start(TaskRequest(instruction="open the drawer"))

    assert error.value.running_task_id == task.id


async def test_cancel_ends_the_task_and_frees_the_vla() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue()))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        cancelled = await runner.cancel(task.id)
        assert await _next_statuses(events, 2) == [TaskRunning(), TaskCancelled()]

    assert cancelled.status == TaskCancelled()
    assert (await runner.start(PICK_CUP)).status == TaskRunning()


async def test_cancel_after_the_end_keeps_the_final_status() -> None:
    commands = asyncio.Queue[str]()
    runner = TaskRunner(TerminalPolicy(commands))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        commands.put_nowait("s")
        await _next_statuses(events, 2)

    assert (await runner.cancel(task.id)).status == TaskSucceeded()


async def test_unknown_task_is_an_error() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue()))
    task_id = UUID("6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11")

    with pytest.raises(UnknownTaskError):
        await runner.get(task_id)


async def test_random_policy_is_reproducible_with_a_seed() -> None:
    config = RandomPolicyConfig(min_duration=0, max_duration=0, failure_rate=0.5, seed=7)
    runner = TaskRunner(RandomPolicy(config))
    finals = []

    async with runner.events() as events:
        for _ in range(4):
            await runner.start(PICK_CUP)
            finals.append((await _next_statuses(events, 2))[1])

    assert finals == [
        TaskFailed(reason="the object is out of reach"),
        TaskSucceeded(),
        TaskFailed(reason="the arm hit a joint limit"),
        TaskFailed(reason="the path is blocked"),
    ]


def test_random_policy_rejects_inverted_durations() -> None:
    with pytest.raises(
        pydantic.ValidationError, match="min_duration 5.0 is above max_duration 1.0"
    ):
        RandomPolicyConfig(min_duration=5, max_duration=1, failure_rate=0, seed=None)
