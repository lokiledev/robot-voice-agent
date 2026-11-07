from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from rva_schemas.vision import FaceEvent, KnownPerson, Remembered, TrackedFace


class Vision(Protocol):
    """Who the robot sees, and its memory of the people it knows.

    The memory holds their names, faces, and which of them other operators have authorized.
    """

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

    async def rename(self, track_id: int, name: str) -> KnownPerson:
        """Change the name of the person on a track, keeping who they are.

        Raises:
            UnknownTrackError: If no face with this track is in view.
            NotRecognizedError: If the face on this track is nobody known yet.
            NameTakenError: If someone else already has this name.
        """
        ...

    async def people(self) -> tuple[KnownPerson, ...]:
        """Everyone known, in view or not."""
        ...

    def granted_operators(self) -> frozenset[UUID]:
        """Operators authorized by another operator, besides those listed in the profile."""
        ...

    async def grant_operator(self, user_id: UUID, granted_by: UUID) -> None: ...

    async def revoke_operator(self, user_id: UUID) -> bool:
        """Remove an operator granted by another one. Returns whether they were one."""
        ...
