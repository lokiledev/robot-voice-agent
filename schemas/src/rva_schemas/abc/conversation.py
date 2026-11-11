from collections.abc import AsyncIterator, Callable
from typing import Protocol

from rva_schemas.dialogue import Spoken, UserUtterance


class UserInput(Protocol):
    """Where the agent hears the user: the voice pipeline, or a terminal in text mode."""

    def utterances(self) -> AsyncIterator[UserUtterance]:
        """Each completed user turn, including turns that talk over the robot."""
        ...


class SpeechOutput(Protocol):
    """How the agent talks to the user: speech synthesis, or a terminal in text mode."""

    async def say(self, sentences: AsyncIterator[str], keep_going: Callable[[], bool]) -> Spoken:
        """Speak sentences as they arrive and return once done or cut off.

        When the user talks over the robot, playback stops, `sentences` is no longer consumed, and
        the result holds only what was heard. `keep_going` is asked before each sentence: once it
        says no, e.g. because the listener is gone, the robot stops there, cut short.
        """
        ...

    def interrupt(self) -> None:
        """Cut off what the robot is saying, as when the user talks over it."""
        ...
