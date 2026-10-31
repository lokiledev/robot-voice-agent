import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Final

from rva_schemas.dialogue import Spoken, UserUtterance


class TerminalUserInput:
    """`UserInput` reading one utterance per line typed in the terminal."""

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        async for line in read_lines():
            if line.strip():
                now = datetime.now(UTC)
                yield UserUtterance(text=line.strip(), started_at=now, ended_at=now)


class TerminalSpeechOutput:
    """`SpeechOutput` writing what the robot says to the terminal, sentence by sentence."""

    def __init__(self, robot_name: str) -> None:
        self._robot_name: Final = robot_name

    async def say(self, sentences: AsyncIterator[str]) -> Spoken:
        said: list[str] = []
        async for sentence in sentences:
            write(f"{self._robot_name}: {sentence}" if not said else f" {sentence}")
            said.append(sentence)
        if said:
            write("\n")
        return Spoken(text=" ".join(said), interrupted=False)


def write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


async def read_lines() -> AsyncGenerator[str]:
    """Lines typed on stdin until end of input, without a thread blocked on reading."""
    lines = asyncio.Queue[str | None]()
    loop = asyncio.get_running_loop()

    def on_readable() -> None:
        line = sys.stdin.readline()
        lines.put_nowait(line or None)
        if not line:
            loop.remove_reader(sys.stdin)

    loop.add_reader(sys.stdin, on_readable)
    try:
        while (line := await lines.get()) is not None:
            yield line
    finally:
        loop.remove_reader(sys.stdin)
