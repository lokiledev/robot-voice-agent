import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import Final

from rva_schemas.dialogue import LlmOutput, Message, Spoken, ToolSpec, UserUtterance
from rva_schemas.record import Record


class ScriptedLanguageModel:
    """Answers each call with the next scripted turn and keeps the messages it was given."""

    def __init__(self, turns: Sequence[Sequence[LlmOutput]]) -> None:
        self._turns: Final = list(turns)
        self.requests: Final[list[tuple[Message, ...]]] = []

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncGenerator[LlmOutput]:
        self.requests.append(tuple(messages))
        if not self._turns:
            raise AssertionError(f"Unexpected LLM call after: {messages[-1]!r}")
        for output in self._turns.pop(0):
            yield output


class QueueUserInput:
    def __init__(self) -> None:
        self._lines: Final = asyncio.Queue[str | None]()

    def say(self, text: str) -> None:
        self._lines.put_nowait(text)

    def leave(self) -> None:
        self._lines.put_nowait(None)

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        while (text := await self._lines.get()) is not None:
            now = datetime.now(UTC)
            yield UserUtterance(text=text, started_at=now, ended_at=now)


class RecordingSpeech:
    """Collects what the robot says; interrupts it after a number of sentences if asked to."""

    def __init__(self, interrupt_after: int | None) -> None:
        self._interrupt_after: Final = interrupt_after
        self.said: Final[list[str]] = []
        self._changed: Final = asyncio.Condition()

    async def say(self, sentences: AsyncIterator[str]) -> Spoken:
        heard: list[str] = []
        interrupted = False
        async for sentence in sentences:
            heard.append(sentence)
            if len(heard) == self._interrupt_after:
                interrupted = True
                break
        async with self._changed:
            self.said.extend(heard)
            self._changed.notify_all()
        return Spoken(text=" ".join(heard), interrupted=interrupted)

    async def wait_for(self, sentence: str) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: sentence in self.said)


class ListRecorder:
    def __init__(self) -> None:
        self.entries: Final[list[Record]] = []
        self._changed: Final = asyncio.Condition()

    def record(self, entry: Record) -> None:
        self.entries.append(entry)
        asyncio.get_running_loop().create_task(self._notify())

    async def wait_until(self, predicate: Callable[[list[Record]], bool]) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: predicate(self.entries))

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()
