from collections.abc import AsyncIterator
from typing import Protocol

from rva_schemas.sound import HeardDirection


class SoundDirections(Protocol):
    """A microphone array that tells where it hears speech from: a direction-of-arrival sensor."""

    def directions(self) -> AsyncIterator[HeardDirection]:
        """Where speech is heard from, as the array's frame has it, as it comes, until cancelled."""
        ...
