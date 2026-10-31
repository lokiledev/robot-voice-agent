import asyncio
import contextlib
import os
import sys
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, assert_never
from urllib.parse import urlsplit

import cyclopts
import httpx
from loguru import logger

from rva_agent.agent import Agent
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import split_sentences
from rva_agent.tools import TaskTools
from rva_app.config import RuntimeConfig, load_admins
from rva_app.page import MergedInput, Recorders, serve_page
from rva_app.servers import Server, ServersUnavailableError, require_servers
from rva_app.services import Service, ServiceFailedError, services_running
from rva_app.terminal import TerminalSpeechOutput, TerminalUserInput, read_lines, write
from rva_app.voice import open_voice_input, open_voice_output, robot_speech
from rva_body_client.client import HttpBodyClient
from rva_llm.client import OpenAiLanguageModel
from rva_llm.server import llama_server_command
from rva_observability.session import start_session
from rva_schemas.abc.conversation import SpeechOutput, UserInput
from rva_schemas.abc.recorder import Recorder
from rva_schemas.dialogue import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextDelta,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    UserUtterance,
)
from rva_schemas.profile import Profile
from rva_schemas.vision import UnknownTrackError
from rva_ui.media import (
    MEDIA_SERVER_ENVIRONMENT,
    MediaServerConfig,
    install_media_server,
    media_server_command,
    media_server_directory,
)
from rva_ui.server import UiServer
from rva_vision_faces.config import FaceVisionConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.vision import FaceVision
from rva_vla_client.client import HttpVlaClient

app = cyclopts.App(
    name="lune",
    help="Lune, a local voice and vision agent for a humanoid robot. Without a command, runs\n    everything; the commands try each part on its own.",
)

type ProfileDir = Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")]
_DEFAULT_PROFILE = Path("config/default")


@app.command
def prompt(*, profile: ProfileDir = _DEFAULT_PROFILE, camera: bool = True) -> None:
    """Print the system prompt rendered from a profile.

    Args:
        profile: Robot profile directory.
        camera: Include what the robot is told about the people it sees.
    """
    write(render_system_prompt(Profile.load(profile), vision=camera))


@app.command
def serve_llm(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """Run llama-server in the foreground with the profile's settings."""
    command = llama_server_command(RuntimeConfig.load(profile).llm_server)
    os.execv(command[0], command)


@app.command
def serve_video(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """Run MediaMTX in the foreground, to show the camera on the session page.

    It is downloaded from its GitHub releases on first use.
    """
    config = RuntimeConfig.load(profile).media_server
    command = media_server_command(_media_server_executable(config))
    # It writes a self-signed certificate where it runs.
    os.chdir(media_server_directory(config))
    os.execve(command[0], command, {**os.environ, **MEDIA_SERVER_ENVIRONMENT})


@app.default
def everything(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    text: bool = False,
    camera: bool = True,
    viewer: bool = False,
    show: bool = False,
    video: bool = False,
) -> None:
    """Run everything: the servers the agent needs, then the agent with its session page.

    Starts the LLM server, the mock robot and, with the camera, MediaMTX, unless they already
    run. Their output goes to `servers/` in the captures directory, and they stop with the agent.

    Args:
        profile: Robot profile directory.
        text: Type to the robot instead of talking.
        camera: Watch for people with the camera.
        viewer: Stream the recording to a rerun viewer.
        show: Also show the camera with the tracked faces in a window.
        video: Record the camera frames with the session, to replay them in rerun.
    """
    _run(_up(profile, text, camera, viewer, show, video))


@app.command
def llm(*, profile: ProfileDir = _DEFAULT_PROFILE, system_prompt: bool = True) -> None:
    """Chat with the LLM alone: tool calls are shown, not run.

    Args:
        profile: Robot profile directory.
        system_prompt: Send the agent's system prompt and tools.
    """
    _run(_chat_with_llm(profile, system_prompt))


@app.command
def text(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    viewer: bool = False,
    camera: bool = True,
    show: bool = False,
    video: bool = False,
    ui: bool = False,
) -> None:
    """Run the agent in the terminal: type to talk, read its replies. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
        video: Record the camera frames with the session, to replay them in rerun.
        ui: Show the session on a web page, where you can type too. The camera needs MediaMTX.
    """
    _run(_run_text_agent(profile, viewer, camera, show, video, ui))


@app.command
def listen(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Print what the microphone hears, turn by turn: VAD, end of turn and ASR, without the agent.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the VAD probability, to a rerun viewer.
    """
    asyncio.run(_listen(profile, viewer))


@app.command
def say(*, profile: ProfileDir = _DEFAULT_PROFILE, voice: str | None = None) -> None:
    """Speak each line typed, sentence by sentence.

    Args:
        profile: Robot profile directory.
        voice: Kokoro voice to try instead of the profile's, e.g. af_heart or bf_emma.
    """
    asyncio.run(_say(profile, voice))


@app.command
def run(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    viewer: bool = False,
    camera: bool = True,
    show: bool = False,
    video: bool = False,
    ui: bool = False,
) -> None:
    """Run the voice agent. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
        video: Record the camera frames with the session, to replay them in rerun.
        ui: Show the session on a web page, where you can type too. The camera needs MediaMTX.
    """
    _run(_run_voice_agent(profile, viewer, camera, show, video, ui))


def _run(main: Coroutine[object, object, None]) -> None:
    try:
        asyncio.run(main)
    except (ServersUnavailableError, ServiceFailedError) as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error


async def _up(
    profile_dir: Path, text: bool, camera: bool, viewer: bool, show: bool, video: bool
) -> None:
    config = RuntimeConfig.load(profile_dir)
    logs = config.observability.captures_dir.expanduser() / "servers"
    async with services_running(_services(config, camera), logs):
        write(f"Session page: http://{config.ui.host}:{config.ui.port}\n")
        run_agent = _run_text_agent if text else _run_voice_agent
        await run_agent(profile_dir, viewer, camera, show, video, True)


def _services(config: RuntimeConfig, camera: bool) -> list[Service]:
    """The servers `lune` runs: MediaMTX only for the camera."""
    llm = config.llm_server
    robot = urlsplit(str(config.vla.base_url))
    assert robot.hostname is not None and robot.port is not None, "validated as an HTTP URL"
    mock = Path(sys.executable).with_name("rva-vla-mock")
    services = [
        Service(
            # Its health check fails until the model is loaded, unlike its other routes.
            server=Server(
                name="LLM server",
                url=f"http://{llm.host}:{llm.port}/health",
                how_to_start="start it with `uv run lune serve-llm`",
                ready_when="succeeds",
            ),
            command=lambda: llama_server_command(llm),
            environment={},
            directory=None,
            startup_s=300,
        ),
        Service(
            server=_body_server(config),
            command=lambda: [str(mock), "--host", robot.hostname, "--port", str(robot.port)],
            environment={},
            directory=None,
            startup_s=30,
        ),
    ]
    if camera:
        services.append(
            Service(
                server=_media_server(config),
                command=lambda: media_server_command(_media_server_executable(config.media_server)),
                environment=MEDIA_SERVER_ENVIRONMENT,
                # It writes a self-signed certificate where it runs.
                directory=media_server_directory(config.media_server),
                startup_s=30,
            )
        )
    return services


def _media_server_executable(config: MediaServerConfig) -> Path:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        return install_media_server(config, client)


def _llm_server(config: RuntimeConfig) -> Server:
    return Server(
        name="LLM server",
        url=f"{config.llm.base_url}",
        how_to_start="start it with `uv run lune serve-llm`",
        ready_when="answers",
    )


def _vla_server(config: RuntimeConfig) -> Server:
    return Server(
        name="VLA",
        url=f"{config.vla.base_url}",
        how_to_start="start the mock with `uv run rva-vla-mock --interactive`, or set vla.base_url in runtime.yaml",
        ready_when="answers",
    )


def _body_server(config: RuntimeConfig) -> Server:
    return Server(
        name="Body API",
        url=f"{str(config.body.base_url).rstrip('/')}/body/state",
        how_to_start="start the mock with `uv run rva-vla-mock --interactive`; a mock started"
        " before the body API existed must be restarted",
        ready_when="route_exists",
    )


def _media_server(config: RuntimeConfig) -> Server:
    return Server(
        name="MediaMTX",
        url=config.ui.video_page,
        how_to_start="start it with `uv run lune serve-video`, or run with --no-camera",
        ready_when="answers",
    )


def _agent_servers(config: RuntimeConfig, streamed: bool) -> list[Server]:
    """The servers the agent needs; MediaMTX too if the camera is `streamed`."""
    servers = [_llm_server(config), _vla_server(config), _body_server(config)]
    return [*servers, _media_server(config)] if streamed else servers


def _page(
    config: RuntimeConfig, ui: bool, camera: bool
) -> AbstractAsyncContextManager[UiServer | None]:
    if not ui:
        return contextlib.nullcontext(None)
    return serve_page(config.ui, config.ui.video_page if camera else None)


def _recorder(session: Recorder, page: UiServer | None) -> Recorder:
    return session if page is None else Recorders(session, page)


def _typed_too(user_input: UserInput, page: UiServer | None) -> UserInput:
    """`user_input`, and the lines typed on the page if any."""
    return user_input if page is None else MergedInput(user_input, page)


async def _listen(profile_dir: Path, viewer: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    recorder = start_session(config.observability, "listen", viewer=viewer, console_level="WARNING")
    # Nothing is played here, so the robot never speaks and no transcript is an echo.
    silent = robot_speech(config.audio)
    async with open_voice_input(config.audio, recorder, silent, lambda: write("… ")) as voice:
        write("Listening. Ctrl+C to stop.\n")
        async for utterance in voice.utterances():
            delay_ms = (datetime.now(UTC) - utterance.ended_at).total_seconds() * 1000
            write(f"{utterance.text}  [{delay_ms:.0f} ms after the end of the turn]\n")


async def _say(profile_dir: Path, voice_name: str | None) -> None:
    config = RuntimeConfig.load(profile_dir)
    recorder = start_session(config.observability, "say", viewer=False, console_level="WARNING")
    tts = config.tts if voice_name is None else config.tts.model_copy(update={"voice": voice_name})
    speaking = robot_speech(config.audio)
    async with open_voice_output(config.audio, tts, speaking, recorder) as voice:
        write("Type sentences to hear them. Ctrl+D to stop.\n")
        async for line in read_lines():
            started = time.perf_counter()
            spoken = await voice.say(split_sentences(_one(line)))
            write(f"[{(time.perf_counter() - started):.1f} s] {spoken.text}\n")


async def _run_voice_agent(
    profile_dir: Path, viewer: bool, camera: bool, show: bool, video: bool, ui: bool
) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(*_agent_servers(config, streamed=ui and camera))
    session = start_session(config.observability, "voice", viewer=viewer, console_level="INFO")
    speaking = robot_speech(config.audio)
    async with (
        _page(config, ui, camera) as page,
        HttpVlaClient(config.vla) as vla,
        HttpBodyClient(config.body) as body,
    ):
        recorder = _recorder(session, page)
        vision = (
            await asyncio.to_thread(
                _face_vision,
                config.vision,
                recorder,
                video,
                stream_video=page is not None,
                show_window=show,
                on_key=None,
            )
            if camera
            else None
        )
        async with (
            open_voice_output(config.audio, config.tts, speaking, recorder) as speech,
            open_voice_input(config.audio, recorder, speaking, speech.interrupt) as voice,
        ):
            await _run_agent(
                profile,
                config,
                vla,
                body,
                speech,
                recorder,
                _typed_too(voice, page),
                vision,
            )


async def _run_agent(
    profile: Profile,
    config: RuntimeConfig,
    vla: HttpVlaClient,
    body: HttpBodyClient,
    speech: SpeechOutput,
    recorder: Recorder,
    user_input: UserInput,
    vision: FaceVision | None,
) -> None:
    agent = Agent(
        llm=OpenAiLanguageModel(config.llm),
        vla=vla,
        body=body,
        user_input=_captioned(user_input, vision),
        speech=speech,
        recorder=recorder,
        vision=vision,
        profile=profile,
        admins=load_admins(config.admins),
        config=config.agent,
    )
    logger.info("{} is up", profile.character.name)
    await _with_vision(agent.run(), vision)


async def _one(text: str) -> AsyncGenerator[str]:
    yield text


@app.command
def see(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    viewer: bool = False,
    video: bool = False,
    stream: bool = False,
) -> None:
    """Show the camera with each tracked face and who it is, without the agent.

    Faces coming and going are logged. Press n in the window to type a name for the nearest face in
    the terminal, q to quit.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the per-frame timings, to a rerun viewer.
        video: Record the camera frames with the session, to replay them in rerun.
        stream: Also publish the window's image to MediaMTX, to watch on its WebRTC page.
    """
    _run(_see(profile, viewer, video, stream))


async def _run_text_agent(
    profile_dir: Path, viewer: bool, camera: bool, show: bool, video: bool, ui: bool
) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(*_agent_servers(config, streamed=ui and camera))
    session = start_session(config.observability, "text", viewer=viewer, console_level="WARNING")
    async with (
        _page(config, ui, camera) as page,
        HttpVlaClient(config.vla) as vla,
        HttpBodyClient(config.body) as body,
    ):
        recorder = _recorder(session, page)
        vision = (
            await asyncio.to_thread(
                _face_vision,
                config.vision,
                recorder,
                video,
                stream_video=page is not None,
                show_window=show,
                on_key=None,
            )
            if camera
            else None
        )
        await _run_agent(
            profile,
            config,
            vla,
            body,
            TerminalSpeechOutput(profile.character.name, recorder),
            recorder,
            _typed_too(TerminalUserInput(), page),
            vision,
        )


@app.command
def people(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """List the user id and name of everyone the robot knows, marking admins and operators."""
    config = RuntimeConfig.load(profile)
    database = _face_database(config.vision)
    admins = load_admins(config.admins).user_ids
    granted = database.granted_operators()
    known = database.people()
    if not known:
        write("Nobody is known yet. Name a face with `lune see`, then n.\n")
    for person in known:
        role = (
            "  admin"
            if person.user_id in admins
            else "  operator"
            if person.user_id in granted
            else ""
        )
        write(f"{person.user_id}  {person.name}{role}\n")


async def _see(profile_dir: Path, viewer: bool, video: bool, stream: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    if stream:
        await require_servers(_media_server(config))
    recorder = start_session(config.observability, "see", viewer=viewer, console_level="INFO")
    keys = asyncio.Queue[str]()
    vision = await asyncio.to_thread(
        _face_vision,
        config.vision,
        recorder,
        video,
        stream_video=stream,
        show_window=True,
        on_key=keys.put_nowait,
    )
    if stream:
        logger.info("Streaming the window to {}", config.ui.video_page)
    async with asyncio.TaskGroup() as group:
        naming = group.create_task(_name_nearest_face(vision, keys))
        await vision.run()
        naming.cancel()


async def _name_nearest_face(vision: FaceVision, keys: asyncio.Queue[str]) -> None:
    async with aclosing(read_lines()) as lines:
        while True:
            if await keys.get() != "n":
                continue
            nearest = next((face for face in vision.faces() if face.nearest), None)
            if nearest is None:
                logger.warning("No face is in view to name")
                continue
            write(f"Name of track {nearest.track_id}: ")
            line = await anext(lines, None)
            if line is None:
                return
            if not line.strip():
                continue
            try:
                remembered = await vision.remember(nearest.track_id, line.strip())
            except UnknownTrackError:
                logger.warning("Track {} left before it was named", nearest.track_id)
                continue
            if remembered.previous is not None:
                logger.info(
                    "Track {} was recognized as {}", nearest.track_id, remembered.previous.name
                )
            write(f"{remembered.person.name} is user {remembered.person.user_id}\n")


def _face_vision(
    config: FaceVisionConfig,
    recorder: Recorder,
    record_video: bool,
    *,
    stream_video: bool,
    show_window: bool,
    on_key: Callable[[str], None] | None,
) -> FaceVision:
    return FaceVision(
        config,
        _face_database(config),
        recorder=recorder,
        record_video=record_video,
        stream_video=stream_video,
        show_window=show_window,
        on_key=on_key,
    )


def _captioned(user_input: UserInput, vision: FaceVision | None) -> UserInput:
    """`user_input`, showing each sentence over the camera image too."""
    if vision is None:
        return user_input
    return _CaptionedInput(user_input, vision)


class _CaptionedInput:
    def __init__(self, user_input: UserInput, vision: FaceVision) -> None:
        self._user_input = user_input
        self._vision = vision

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        async for utterance in self._user_input.utterances():
            self._vision.caption(utterance.text)
            yield utterance


def _face_database(config: FaceVisionConfig) -> FaceDatabase:
    return FaceDatabase(config.database.expanduser(), config.max_embeddings_per_person)


async def _with_vision(main: Coroutine[None, None, None], vision: FaceVision | None) -> None:
    """Run `main` while the camera is watched.

    A camera failure stops `main`; closing the window stops only the camera.
    """
    if vision is None:
        await main
        return
    async with asyncio.TaskGroup() as group:
        seeing = group.create_task(vision.run())
        await main
        seeing.cancel()


async def _chat_with_llm(profile_dir: Path, system_prompt: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config))
    model = OpenAiLanguageModel(config.llm)
    tools = TaskTools.specs if system_prompt else ()
    messages: list[Message] = []
    if system_prompt:
        messages.append(
            SystemMessage(content=render_system_prompt(Profile.load(profile_dir), vision=False))
        )
    async for line in read_lines():
        messages.append(UserMessage(content=line.strip()))
        text: list[str] = []
        calls: list[ToolCall] = []
        async for output in model.stream(messages, tools):
            match output:
                case TextDelta():
                    write(output.text)
                    text.append(output.text)
                case ToolCall():
                    write(f"\n[tool call] {output.name}({output.arguments})")
                    calls.append(output)
                case _:
                    assert_never(output)
        write("\n")
        messages.append(AssistantMessage(content="".join(text), tool_calls=tuple(calls)))
        messages.extend(
            ToolResultMessage(tool_call_id=call.id, content='{"status": "not run in lune llm"}')
            for call in calls
        )
