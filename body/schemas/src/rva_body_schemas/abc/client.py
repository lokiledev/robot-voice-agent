from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from rva_body_schemas.motion import BodyState, Motion, MotionEvent, MotionRequest


class BodyClient(Protocol):
    """The robot's body besides the VLA: walking, animations and where it looks."""

    async def start(self, request: MotionRequest) -> Motion:
        """Start a motion and return once it began, without waiting for it to end.

        Raises:
            BodyBusyError: If the body is already moving or running a VLA task.
        """
        ...

    async def cancel(self, motion_id: UUID) -> Motion:
        """Stop a motion. Its end arrives through `events`.

        Raises:
            UnknownMotionError: If the body has no motion with this id.
        """
        ...

    async def stay(self) -> None:
        """Stop walking or animating, if doing so, and stay where it is."""
        ...

    async def look_at(self, user_id: UUID) -> None:
        """Keep turning the head and body towards a person until told otherwise or walking."""
        ...

    async def stop_looking(self) -> None: ...

    async def state(self) -> BodyState: ...

    def events(self) -> AbstractAsyncContextManager[AsyncIterator[MotionEvent]]:
        """Subscribe to status changes of every motion; every event after entering is delivered."""
        ...
