import asyncio
import socket
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, Literal, assert_never

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import StringConstraints

from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import UserUtterance
from rva_schemas.record import (
    AgentStarted,
    CameraFrame,
    FaceUpdate,
    GazeUpdate,
    Measurement,
    MotionUpdate,
    MuteUpdate,
    PeopleInView,
    Record,
    RobotLine,
    RobotSentence,
    TaskUpdate,
    ToolInvocation,
    UserLine,
)

_PAGE: Final = Path(__file__).parent / "page"
_BACKLOG: Final = 5000
"""Entries replayed to a page that opens, so that a reload shows the session so far."""
_SHUTDOWN_TIMEOUT_S: Final = 1


class UiConfig(RVABaseModel):
    host: str
    """`127.0.0.1` to open the page on this machine only."""
    port: int
    video_page: str
    """MediaMTX's WebRTC page for `vision.stream.url`, e.g. `http://127.0.0.1:8889/lune/`."""


class Hello(RVABaseModel):
    """The first message to a page; the session so far follows, then each entry as it comes."""

    kind: Literal["hello"] = "hello"
    video_page: str | None
    """Where to play the camera with the tracked faces, if it is streamed."""


class Entry(RVABaseModel):
    kind: Literal["entry"] = "entry"
    at: datetime
    record: Record


class TypedLine(RVABaseModel):
    """What the page sends: a line typed for the robot."""

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class UiServer:
    """A web page showing a session live, where the user can also type to the robot.

    It is the `Recorder` whose entries go to each open page, and the `UserInput` of the lines typed
    there. Create it in the event loop that serves it; `record` may be called from any thread.
    """

    def __init__(self, video_page: str | None) -> None:
        """Set up the page; `video_page` is where it plays the camera, if the camera is streamed."""
        self._loop: Final = asyncio.get_running_loop()
        self._hello: Final = Hello(video_page=video_page).model_dump_json()
        self._backlog: Final = deque[str](maxlen=_BACKLOG)
        self._pages: Final[set[asyncio.Queue[str]]] = set()
        self._typed: Final = asyncio.Queue[str]()
        # Requests for the page are not latencies of the robot's.
        self._app: Final = FastAPI(telemetry={"tracing": False, "metrics": False, "logs": False})
        self._app.middleware("http")(_revalidate)
        self._app.add_api_websocket_route("/session", self._session)
        self._app.mount("/", StaticFiles(directory=_PAGE, html=True))

    def record(self, entry: Record) -> None:
        match entry:
            case PeopleInView() | CameraFrame():
                # The boxes are drawn into the video stream instead.
                return
            case Measurement():
                replayed = False
            case (
                AgentStarted()
                | UserLine()
                | RobotSentence()
                | RobotLine()
                | ToolInvocation()
                | TaskUpdate()
                | MotionUpdate()
                | GazeUpdate()
                | MuteUpdate()
                | FaceUpdate()
            ):
                replayed = True
            case _:
                assert_never(entry)
        message = Entry(at=datetime.now(UTC), record=entry).model_dump_json()
        self._loop.call_soon_threadsafe(self._publish, message, replayed)

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        """The lines typed on the page, forever."""
        while True:
            text = await self._typed.get()
            now = datetime.now(UTC)
            yield UserUtterance(text=text, started_at=now, ended_at=now)

    async def serve(self, sock: socket.socket) -> None:
        """Serve the page on `sock`, already listening, until cancelled."""
        server = uvicorn.Server(
            uvicorn.Config(
                self._app, log_level="warning", timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT_S
            )
        )
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            await asyncio.shield(serving)
        finally:
            server.should_exit = True
            await serving

    def _publish(self, message: str, replayed: bool) -> None:
        if replayed:
            self._backlog.append(message)
        for page in self._pages:
            page.put_nowait(message)

    async def _session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        queue = asyncio.Queue[str]()
        for message in (self._hello, *self._backlog):
            queue.put_nowait(message)
        self._pages.add(queue)
        forwarding = asyncio.create_task(_forward(queue, websocket))
        try:
            async for text in websocket.iter_text():
                self._typed.put_nowait(TypedLine.model_validate_json(text).text)
        finally:
            self._pages.discard(queue)
            forwarding.cancel()


async def _revalidate(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # A browser otherwise keeps the page's files of an older version after a restart.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


async def _forward(queue: asyncio.Queue[str], websocket: WebSocket) -> None:
    try:
        while True:
            await websocket.send_text(await queue.get())
    except WebSocketDisconnect:
        # The page closed; its session ends with it.
        return
