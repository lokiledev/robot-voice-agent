from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from rva_schemas.vision import FaceEvent, KnownPerson, Remembered, TrackedFace


class Vision(Protocol):
    """Who the robot sees, and the memory of the names of their faces."""

    def events(self) -> AbstractAsyncContextManager[AsyncIterator[FaceEvent]]:
        """Subscribe to faces appearing and leaving; every event after entering is delivered."""
        ...

    def faces(self) -> tuple[TrackedFace, ...]:
        """The faces in view now."""
        ...

    async def remember(self, track_id: int, name: str) -> Remembered:
        """Associate the face on a track with a name, creating the person if the name is new.

        Raises:
            UnknownTrackError: If no face with this track is in view.
        """
        ...

    async def forget(self, name: str) -> KnownPerson:
        """Delete a person and every sample of their face.

        Raises:
            UnknownPersonError: If nobody has this name.
        """
        ...
