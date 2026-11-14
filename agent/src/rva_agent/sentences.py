import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Final

_SENTENCE_END: Final = re.compile(r"(?<=[.!?;:])\s+|\n+")


async def split_sentences(chunks: AsyncIterator[str]) -> AsyncGenerator[str]:
    """Regroup streamed text into sentences, yielding each as soon as it is complete."""
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        while end := _SENTENCE_END.search(buffer):
            sentence, buffer = buffer[: end.start()].strip(), buffer[end.end() :]
            if sentence:
                yield sentence
    if buffer.strip():
        yield buffer.strip()


class StageDirections:
    """Takes stage directions, like "*waves*", out of a reply as it streams in.

    The speech synthesizer would read them out, asterisks and all, and once one is in the history
    the LLM writes one in every reply after. Everything between two asterisks goes, emphasis
    too: "I *love* it" is said "I  it".
    """

    def __init__(self) -> None:
        self._inside = False
        self.removed: list[str] = []
        """The stage directions taken out so far, without their asterisks."""

    def strip(self, chunk: str) -> str:
        """The next `chunk` of the reply, without what is between asterisks."""
        kept: list[str] = []
        for index, part in enumerate(chunk.split("*")):
            if index > 0:
                self._inside = not self._inside
                if self._inside:
                    self.removed.append("")
            if self._inside:
                self.removed[-1] += part
            else:
                kept.append(part)
        return "".join(kept)
