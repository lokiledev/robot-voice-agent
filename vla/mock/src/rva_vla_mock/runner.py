import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final
from uuid import UUID, uuid4

from loguru import logger

from rva_vla_mock.policies import OutcomePolicy
from rva_vla_schemas.errors import UnknownTaskError, VlaBusyError
from rva_vla_schemas.task import (
    Task,
    TaskCancelled,
    TaskEvent,
    TaskRequest,
    TaskRunning,
    TaskStatus,
)


class TaskRunner:
    """In-process VLA: runs one task at a time and lets a policy decide how each one ends.

    Implements `VlaClient`, so it also serves as an in-process VLA for the agent.
    """

    def __init__(self, policy: OutcomePolicy) -> None:
        self._policy: Final = policy
        self._tasks: Final[dict[UUID, Task]] = {}
        self._active: tuple[UUID, asyncio.Task[None]] | None = None
        self._subscribers: Final[set[asyncio.Queue[TaskEvent]]] = set()

    async def start(self, request: TaskRequest) -> Task:
        if self._active is not None:
            raise VlaBusyError(self._active[0])
        task_id = uuid4()
        logger.info("Task {} started: {!r}", task_id, request.instruction)
        task = self._update(task_id, request.instruction, TaskRunning())
        self._active = (task_id, asyncio.create_task(self._run(task)))
        return task

    async def cancel(self, task_id: UUID) -> Task:
        task = await self.get(task_id)
        if self._active is None or self._active[0] != task_id:
            return task
        self._active[1].cancel()
        self._active = None
        logger.info("Task {} cancelled", task_id)
        return self._update(task_id, task.instruction, TaskCancelled())

    async def get(self, task_id: UUID) -> Task:
        if task_id not in self._tasks:
            raise UnknownTaskError(task_id)
        return self._tasks[task_id]

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[TaskEvent]]:
        queue = asyncio.Queue[TaskEvent]()
        self._subscribers.add(queue)
        try:
            yield _drain(queue)
        finally:
            self._subscribers.discard(queue)

    async def _run(self, task: Task) -> None:
        status = await self._policy.resolve(task)
        self._active = None
        logger.info("Task {} ended: {}", task.id, status.state)
        self._update(task.id, task.instruction, status)

    def _update(self, task_id: UUID, instruction: str, status: TaskStatus) -> Task:
        task = Task(id=task_id, instruction=instruction, status=status)
        self._tasks[task_id] = task
        event = TaskEvent(task_id=task_id, status=status, timestamp=datetime.now(UTC))
        for queue in self._subscribers:
            queue.put_nowait(event)
        return task


async def _drain(queue: asyncio.Queue[TaskEvent]) -> AsyncIterator[TaskEvent]:
    while True:
        yield await queue.get()
