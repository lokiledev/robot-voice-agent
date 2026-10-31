from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from rva_vla_schemas.task import Task, TaskEvent, TaskRequest


class VlaClient(Protocol):
    """The robot's VLA policy, which runs one short-horizon task at a time."""

    async def start(self, request: TaskRequest) -> Task:
        """Start a task and return once the VLA accepted it, without waiting for it to finish.

        Raises:
            VlaBusyError: If another task is running.
        """
        ...

    async def cancel(self, task_id: UUID) -> Task:
        """Ask the VLA to stop a task. The final status arrives through `events`.

        Raises:
            UnknownTaskError: If the VLA has no task with this id.
        """
        ...

    async def get(self, task_id: UUID) -> Task:
        """Current state of a task.

        Raises:
            UnknownTaskError: If the VLA has no task with this id.
        """
        ...

    def events(self) -> AbstractAsyncContextManager[AsyncIterator[TaskEvent]]:
        """Subscribe to status changes of every task, in the order the VLA emitted them.

        Every event emitted after the context is entered is delivered, so enter it before calling
        `start` to see the whole life of a task:

        ```python
        async with vla.events() as events:
            task = await vla.start(request)
            async for event in events: ...
        ```
        """
        ...
