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
