import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from rva_app.page import MergedInput
from rva_schemas.dialogue import UserUtterance


class _Queued:
    """`UserInput` of the lines put in it, until `None` is."""

    def __init__(self) -> None:
        self.lines = asyncio.Queue[str | None]()

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        while (text := await self.lines.get()) is not None:
            now = datetime.now(UTC)
            yield UserUtterance(text=text, started_at=now, ended_at=now)


class _Broken:
    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        raise OSError("microphone unplugged")
        yield  # An async generator.


async def test_lines_of_both_come_in_order_until_the_main_input_ends() -> None:
    main, typed = _Queued(), _Queued()
    merged = MergedInput(main, typed).utterances()

    main.lines.put_nowait("Hello")
    first = await anext(merged)
    typed.lines.put_nowait("Wave at me")
    second = await anext(merged)
    main.lines.put_nowait(None)

    assert [first.text, second.text] == ["Hello", "Wave at me"]
    assert [utterance.text async for utterance in merged] == []


async def test_a_failure_of_the_main_input_is_raised() -> None:
    merged = MergedInput(_Broken(), _Queued()).utterances()

    with pytest.raises(OSError, match="microphone unplugged"):
        await anext(merged)
