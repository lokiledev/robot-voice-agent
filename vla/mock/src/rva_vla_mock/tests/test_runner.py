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
        ("f the grasp slipped\n", TaskFailed(cause="error", reason="the grasp slipped")),
        ("f\n", TaskFailed(cause="error", reason="no reason given")),
    ],
)
async def test_terminal_command_ends_the_task(command: str, final: TaskStatus) -> None:
    commands = asyncio.Queue[str]()
    runner: VlaClient = TaskRunner(TerminalPolicy(commands, fallback=None))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        assert await _next_statuses(events, 1) == [TaskRunning()]
        commands.put_nowait("what\n")
        commands.put_nowait(command)
        assert await _next_statuses(events, 1) == [final]

    assert (await runner.get(task.id)).status == final


async def test_start_while_busy_names_the_running_task() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    task = await runner.start(PICK_CUP)

    with pytest.raises(VlaBusyError) as error:
        await runner.start(TaskRequest(instruction="open the drawer"))

    assert error.value.running_task_id == task.id


async def test_cancel_ends_the_task_and_frees_the_vla() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        cancelled = await runner.cancel(task.id)
        assert await _next_statuses(events, 2) == [TaskRunning(), TaskCancelled()]

    assert cancelled.status == TaskCancelled()
    assert (await runner.start(PICK_CUP)).status == TaskRunning()


async def test_cancel_after_the_end_keeps_the_final_status() -> None:
    commands = asyncio.Queue[str]()
    runner = TaskRunner(TerminalPolicy(commands, fallback=None))

    async with runner.events() as events:
        task = await runner.start(PICK_CUP)
        commands.put_nowait("s")
        await _next_statuses(events, 2)

    assert (await runner.cancel(task.id)).status == TaskSucceeded()


async def test_unknown_task_is_an_error() -> None:
    runner = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    task_id = UUID("6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11")

    with pytest.raises(UnknownTaskError):
        await runner.get(task_id)


async def test_random_policy_is_reproducible_with_a_seed() -> None:
    config = RandomPolicyConfig(min_duration=0, max_duration=0, timeout=1, failure_rate=0.5, seed=7)
    runner = TaskRunner(RandomPolicy(config))
    finals = []

    async with runner.events() as events:
        for _ in range(4):
            await runner.start(PICK_CUP)
            finals.append((await _next_statuses(events, 2))[1])

    assert finals == [
        TaskFailed(cause="error", reason="the object is out of reach"),
        TaskSucceeded(),
        TaskFailed(cause="error", reason="the arm hit a joint limit"),
        TaskFailed(cause="error", reason="the path is blocked"),
    ]


def test_random_policy_rejects_inverted_durations() -> None:
    with pytest.raises(
        pydantic.ValidationError, match="min_duration 5.0 is above max_duration 1.0"
    ):
        RandomPolicyConfig(min_duration=5, max_duration=1, timeout=10, failure_rate=0, seed=None)


async def test_a_task_running_past_the_time_limit_times_out() -> None:
    config = RandomPolicyConfig(
        min_duration=0.2, max_duration=0.2, timeout=0.05, failure_rate=0, seed=None
    )
    runner = TaskRunner(RandomPolicy(config))

    async with runner.events() as events:
        await runner.start(PICK_CUP)
        statuses = await _next_statuses(events, 2)

    assert statuses == [
        TaskRunning(),
        TaskFailed(cause="timeout", reason="it was still not done after 0 seconds"),
    ]


async def test_an_untouched_interactive_task_ends_by_itself() -> None:
    fallback = RandomPolicy(
        RandomPolicyConfig(min_duration=0, max_duration=0, timeout=1, failure_rate=0, seed=None)
    )
    runner = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=fallback))

    async with runner.events() as events:
        await runner.start(PICK_CUP)

        assert await _next_statuses(events, 2) == [TaskRunning(), TaskSucceeded()]


async def test_typing_t_times_a_task_out() -> None:
    commands = asyncio.Queue[str]()
    runner = TaskRunner(TerminalPolicy(commands, fallback=None))

    async with runner.events() as events:
        await runner.start(PICK_CUP)
        commands.put_nowait("t")

        assert (await _next_statuses(events, 2))[1] == TaskFailed(
            cause="timeout", reason="it was taking too long"
        )
