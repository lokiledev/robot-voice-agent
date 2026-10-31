import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final
from uuid import UUID, uuid4

from loguru import logger

from rva_body_schemas.errors import BodyBusyError, UnknownMotionError
from rva_body_schemas.motion import (
    BodyState,
    GoToHuman,
    GoToLocation,
    Motion,
    MotionEvent,
    MotionRequest,
    describe,
)
from rva_vla_mock.policies import OutcomePolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.task import Task, TaskCancelled, TaskRunning, TaskStatus


class BodyMock:
    """In-process body: walks and animates one motion at a time, and turns to look at people.

    It shares the body with the VLA: no motion starts while a VLA task runs. Implements
    `BodyClient`, so it also serves as an in-process body for the agent.
    """

    def __init__(
        self, vla: TaskRunner, navigation: OutcomePolicy, animation: OutcomePolicy
    ) -> None:
        self._vla: Final = vla
        self._navigation: Final = navigation
        self._animation: Final = animation
        self._motions: Final[dict[UUID, Motion]] = {}
        self._active: tuple[UUID, asyncio.Task[None]] | None = None
        self._looking_at: UUID | None = None
        self._subscribers: Final[set[asyncio.Queue[MotionEvent]]] = set()

    async def start(self, request: MotionRequest) -> Motion:
        if (task := self._vla.running) is not None:
            raise BodyBusyError(task.instruction)
        if self._active is not None:
            raise BodyBusyError(describe(self._motions[self._active[0]].action))
        action = request.action
        policy = self._animation
        if isinstance(action, GoToLocation | GoToHuman):
            policy = self._navigation
            if self._looking_at is not None:
                logger.info("Stops looking at {} to walk", self._looking_at)
                self._looking_at = None
        motion = self._update(
            Motion(id=uuid4(), action=action, status=TaskRunning()), TaskRunning()
        )
        logger.info("Motion {} started: {}", motion.id, describe(action))
        self._active = (motion.id, asyncio.create_task(self._run(motion, policy)))
        return motion

    async def cancel(self, motion_id: UUID) -> Motion:
        if motion_id not in self._motions:
            raise UnknownMotionError(motion_id)
        motion = self._motions[motion_id]
        if self._active is None or self._active[0] != motion_id:
            return motion
        self._active[1].cancel()
        self._active = None
        logger.info("Motion {} cancelled", motion_id)
        return self._update(motion, TaskCancelled())

    async def stay(self) -> None:
        logger.info("Stays where it is")
        if self._active is not None:
            await self.cancel(self._active[0])

    async def look_at(self, user_id: UUID) -> None:
        logger.info("Looks at {}", user_id)
        self._looking_at = user_id

    async def stop_looking(self) -> None:
        logger.info("Stops looking at {}", self._looking_at)
        self._looking_at = None

    async def state(self) -> BodyState:
        motion = None if self._active is None else self._motions[self._active[0]]
        return BodyState(motion=motion, looking_at=self._looking_at)

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[MotionEvent]]:
        queue = asyncio.Queue[MotionEvent]()
        self._subscribers.add(queue)
        try:
            yield _drain(queue)
        finally:
            self._subscribers.discard(queue)

    async def _run(self, motion: Motion, policy: OutcomePolicy) -> None:
        # Policies resolve tasks; a motion is one for them, described in words.
        task = Task(id=motion.id, instruction=describe(motion.action), status=TaskRunning())
        status = await policy.resolve(task)
        self._active = None
        logger.info("Motion {} ended: {}", motion.id, status.state)
        self._update(motion, status)

    def _update(self, motion: Motion, status: TaskStatus) -> Motion:
        updated = Motion(id=motion.id, action=motion.action, status=status)
        self._motions[motion.id] = updated
        event = MotionEvent(motion_id=motion.id, status=status, timestamp=datetime.now(UTC))
        for queue in self._subscribers:
            queue.put_nowait(event)
        return updated


async def _drain(queue: asyncio.Queue[MotionEvent]) -> AsyncIterator[MotionEvent]:
    while True:
        yield await queue.get()
