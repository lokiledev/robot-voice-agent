from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from rva_schemas.interaction import EngagementEvent


class Engagement(Protocol):
    """Who among the people in view pays attention to the robot."""

    def events(self) -> AbstractAsyncContextManager[AsyncIterator[EngagementEvent]]:
        """Subscribe to people engaging and disengaging; every event after entering is delivered."""
        ...

    def engaged(self) -> frozenset[int]:
        """The tracks of the people engaged now."""
        ...
