import asyncio
import contextlib
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from loguru import logger

from rva_schemas.abc.conversation import UserInput
from rva_schemas.abc.recorder import Recorder
from rva_schemas.dialogue import UserUtterance
from rva_schemas.record import Record
from rva_ui.server import UiConfig, UiServer


@asynccontextmanager
async def serve_page(config: UiConfig, video_page: str | None) -> AsyncGenerator[UiServer]:
    """Serve the session page for the duration of the context.

    Args:
        config: Where to serve it.
        video_page: Where the page plays the camera, if the camera is streamed.

    Raises:
        OSError: If the port is taken.
    """
    sock = socket.create_server((config.host, config.port))
    page = UiServer(video_page)
    serving = asyncio.create_task(page.serve(sock))
    logger.info("Session page on http://{}:{}", config.host, config.port)
    try:
        yield page
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


class Recorders:
    """`Recorder` passing each entry on to several."""

    def __init__(self, *recorders: Recorder) -> None:
        self._recorders: Final = recorders

    def record(self, entry: Record) -> None:
        for recorder in self._recorders:
            recorder.record(entry)


class MergedInput:
    """`UserInput` of a main input and another, e.g. the page, until the main one ends."""

    def __init__(self, main: UserInput, other: UserInput) -> None:
        self._main: Final = main
        self._other: Final = other

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        queue = asyncio.Queue[UserUtterance | None]()

        async def forward_main() -> None:
            try:
                async for utterance in self._main.utterances():
                    queue.put_nowait(utterance)
            finally:
                # Even on failure, which is raised below.
                queue.put_nowait(None)

        async def forward_other() -> None:
            async for utterance in self._other.utterances():
                queue.put_nowait(utterance)

        main = asyncio.create_task(forward_main())
        other = asyncio.create_task(forward_other())
        try:
            while (utterance := await queue.get()) is not None:
                yield utterance
            await main
        finally:
            main.cancel()
            other.cancel()
