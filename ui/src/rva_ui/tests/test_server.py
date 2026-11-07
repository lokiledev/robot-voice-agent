import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from websockets.asyncio.client import connect

from rva_schemas.record import Measurement, PeopleInView, RobotLine, UserLine
from rva_ui.server import UiServer


@asynccontextmanager
async def _serving(ui: UiServer) -> AsyncGenerator[int]:
    """Serve `ui` on a free localhost port, given to the context."""
    # Listening before uvicorn starts queues early connections, so no need to wait for startup.
    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    assert isinstance(port, int)
    serving = asyncio.create_task(ui.serve(sock))
    try:
        yield port
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


async def test_a_page_gets_the_session_so_far_then_each_entry_as_it_comes() -> None:
    ui = UiServer(video_page="http://127.0.0.1:8889/lune/")
    ui.record(UserLine(text="Hello", speaker=None))
    # Too many to replay: only sent to the pages open at the time.
    ui.record(Measurement(name="vad", value=0.9))

    async with _serving(ui) as port, connect(f"ws://127.0.0.1:{port}/session") as page:
        hello, replayed = json.loads(await page.recv()), json.loads(await page.recv())
        # Drawn into the video stream instead.
        ui.record(
            PeopleInView(
                image_width=96, image_height=54, intrinsics=None, people=(), unmatched_bodies=()
            )
        )
        ui.record(RobotLine(text="Hi!", interrupted=False))
        live = json.loads(await page.recv())

    assert hello == {"kind": "hello", "video_page": "http://127.0.0.1:8889/lune/"}
    assert replayed["record"] == {"kind": "user_line", "text": "Hello", "speaker": None}
    assert live["record"] == {"kind": "robot_line", "text": "Hi!", "interrupted": False}


async def test_lines_typed_on_the_page_are_utterances() -> None:
    ui = UiServer(video_page=None)

    async with _serving(ui) as port, connect(f"ws://127.0.0.1:{port}/session") as page:
        await page.send('{"text": "  Wave at me  "}')
        utterance = await anext(ui.utterances())

    assert utterance.text == "Wave at me"


async def test_serves_the_page() -> None:
    ui = UiServer(video_page=None)

    async with _serving(ui) as port, httpx.AsyncClient() as client:
        page = await client.get(f"http://127.0.0.1:{port}/")
        script = await client.get(f"http://127.0.0.1:{port}/app.js")

    assert page.status_code == 200
    assert '<script src="app.js" defer></script>' in page.text
    assert script.status_code == 200
    assert script.headers["Cache-Control"] == "no-cache"
