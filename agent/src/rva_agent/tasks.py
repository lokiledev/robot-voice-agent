from collections.abc import AsyncIterator, Callable
from typing import Final
from uuid import UUID

from loguru import logger

from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.task import FinalTaskStatus, Task, TaskEvent, TaskRequest, TaskRunning


class TaskSupervisor:
    """Keeps track of the one task the robot's body is running."""

    def __init__(self, vla: VlaClient, on_end: Callable[[Task], None]) -> None:
        """Track tasks of `vla`.

        Args:
            vla: The robot's VLA.
            on_end: Called with the final state of each task started here.
        """
        self._vla: Final = vla
        self._on_end: Final = on_end
        self._active: Task | None = None
        # A fast task can end before `start` returns; its end is held here until then.
        self._early_ends: Final[dict[UUID, FinalTaskStatus]] = {}

    @property
    def active(self) -> Task | None:
        return self._active

    async def watch(self, events: AsyncIterator[TaskEvent]) -> None:
        """Follow task events until `events` ends. Run it for the whole session."""
        async for event in events:
            match event.status:
                case TaskRunning():
                    pass
                case status:
                    self._end(event.task_id, status)

    async def start(self, instruction: str) -> Task:
        """Start a task and make it the active one.

        Raises:
            VlaBusyError: If a task is already running.
        """
        task = await self._vla.start(TaskRequest(instruction=instruction))
        self._active = task
        if (status := self._early_ends.pop(task.id, None)) is not None:
            self._end(task.id, status)
        return task

    async def cancel(self) -> Task | None:
        """Cancel the running task, if any. Its end is reported through `on_end`."""
        if self._active is None:
            return None
        return await self._vla.cancel(self._active.id)

    def _end(self, task_id: UUID, status: FinalTaskStatus) -> None:
        if self._active is None or self._active.id != task_id:
            self._early_ends[task_id] = status
            return
        task = Task(id=task_id, instruction=self._active.instruction, status=status)
        self._active = None
        logger.info("Task {!r} ended: {}", task.instruction, status.state)
        self._on_end(task)
