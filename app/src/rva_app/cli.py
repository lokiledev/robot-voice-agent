import asyncio
import contextlib
import os
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, assert_never
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
from rva_attention.attention import VisualAttention
from rva_attention.config import VisualAttentionConfig
from rva_attention.follow import follow
from rva_body_client.client import HttpBodyClient
from rva_interaction.input import AttributedInput
from rva_interaction.interaction import Interaction
from rva_llm.client import OpenAiLanguageModel
from rva_llm.server import llama_server_command
from rva_multimodal.config import ActiveSpeakerConfig
from rva_multimodal.light_asd import open_light_asd
from rva_observability.session import start_session
from rva_schemas.abc.conversation import SpeechOutput, UserInput
from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speaker import SpeakerDetector
from rva_schemas.dialogue import (
    AssistantMessage,
    Message,
    NobodySeenSpeaking,
    SeenSpeaking,
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
from rva_vision_viewer.monitor import CameraMonitor
from rva_vla_client.client import HttpVlaClient
from rva_voice_audio.devices import Microphone

app = cyclopts.App(
    name="lune",
    help="Lune, a local voice and vision agent for a humanoid robot. Without a command, runs\n    everything; the commands try each part on its own.",
)

type ProfileDir = Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")]
_DEFAULT_PROFILE = Path("config/default")
_WINDOW_SIZE: Final = (960, 540)
"""Size of the camera image in the debug window."""


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
    depth: bool = True,
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
        video: Record the camera frames and their depth with the session, to replay them in rerun.
        depth: Estimate the camera's depth, if the profile has a depth model.
    """
    _run(_up(profile, text, camera, viewer, show, video, depth))


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
    depth: bool = True,
) -> None:
    """Run the agent in the terminal: type to talk, read its replies. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
        video: Record the camera frames and their depth with the session, to replay them in rerun.
        ui: Show the session on a web page, where you can type too. The camera needs MediaMTX.
        depth: Estimate the camera's depth, if the profile has a depth model.
    """
    _run(_run_text_agent(profile, viewer, camera, show, video, ui, depth))


@app.command
def listen(
    *, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False, camera: bool = False
) -> None:
    """Print what the microphone hears, turn by turn: VAD, end of turn and ASR, without the agent.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the VAD probability, to a rerun viewer.
        camera: Also print which face in view said each turn, from its lips and the sound.
    """
    _run(_listen(profile, viewer, camera))


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
    depth: bool = True,
) -> None:
    """Run the voice agent. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
        video: Record the camera frames and their depth with the session, to replay them in rerun.
        ui: Show the session on a web page, where you can type too. The camera needs MediaMTX.
        depth: Estimate the camera's depth, if the profile has a depth model.
    """
    _run(_run_voice_agent(profile, viewer, camera, show, video, ui, depth))


def _run(main: Coroutine[object, object, None]) -> None:
    try:
        asyncio.run(main)
    except (ServersUnavailableError, ServiceFailedError) as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error


async def _up(
    profile_dir: Path,
    text: bool,
    camera: bool,
    viewer: bool,
    show: bool,
    video: bool,
    depth: bool,
) -> None:
    config = RuntimeConfig.load(profile_dir)
    logs = config.observability.captures_dir.expanduser() / "servers"
    async with services_running(_services(config, camera), logs):
        write(f"Session page: http://{config.ui.host}:{config.ui.port}\n")
        run_agent = _run_text_agent if text else _run_voice_agent
        await run_agent(profile_dir, viewer, camera, show, video, True, depth)


def _services(config: RuntimeConfig, camera: bool) -> list[Service]:
    """The servers `lune` runs: MediaMTX only for the camera."""
    llm = config.llm_server
    robot = urlsplit(str(config.vla.base_url))
    assert robot.hostname is not None and robot.port is not None, "validated as an HTTP URL"
    mock = Path(sys.executable).with_name("lune-mock")
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
        how_to_start="start the mock with `uv run lune-mock --interactive`, or set vla.base_url in runtime.yaml",
        ready_when="answers",
    )


def _body_server(config: RuntimeConfig) -> Server:
    return Server(
        name="Body API",
        url=f"{str(config.body.base_url).rstrip('/')}/body/state",
        how_to_start="start the mock with `uv run lune-mock --interactive`; a mock started"
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


def _recorder(session: Recorder, page: UiServer | None, monitor: CameraMonitor | None) -> Recorder:
    others = [recorder for recorder in (page, monitor) if recorder is not None]
    return session if not others else Recorders(session, *others)


def _typed_too(user_input: UserInput, page: UiServer | None) -> UserInput:
    """`user_input`, and the lines typed on the page if any."""
    return user_input if page is None else MergedInput(user_input, page)


async def _listen(profile_dir: Path, viewer: bool, camera: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    recorder = start_session(config.observability, "listen", viewer=viewer, console_level="WARNING")
    # Nothing is played here, so the robot never speaks and no transcript is an echo.
    silent = robot_speech(config.audio)
    if camera and config.active_speaker is None:
        raise SystemExit(f"{profile_dir / 'runtime.yaml'} has no active_speaker to tell who speaks")
    active_speaker = config.active_speaker if camera else None
    speaker = await _speaker(active_speaker)
    async with open_voice_input(
        config.audio, config.asr, recorder, silent, lambda: write("… "), _anyone, speaker
    ) as voice:
        write("Listening. Ctrl+C to stop.\n")
        if active_speaker is None:
            await _print_heard(voice)
            return
        vision = await asyncio.to_thread(
            _face_vision, config.vision, recorder, False, depth=True, speaker=speaker
        )
        interaction = Interaction(config.interaction, active_speaker.delay_s, recorder)
        await _with_vision(
            _print_heard(AttributedInput(voice, interaction)),
            vision,
            None,
            config.attention,
            interaction,
            recorder,
        )


def _anyone() -> bool:
    """Whose speech may stop the robot's, where nobody is told apart."""
    return True


async def _print_heard(user_input: UserInput) -> None:
    async for utterance in user_input.utterances():
        delay_ms = (datetime.now(UTC) - utterance.ended_at).total_seconds() * 1000
        match utterance.speaker:
            case None:
                who = ""
            case SeenSpeaking(track_id=track_id, person=person, to_robot=to_robot):
                name = "unknown person" if person is None else person.name
                to = "to the robot" if to_robot else "not to the robot"
                who = f"[{name}, track {track_id}, {to}] "
            case NobodySeenSpeaking():
                who = "[nobody seen speaking] "
            case _:
                assert_never(utterance.speaker)
        write(f"{who}{utterance.text}  [{delay_ms:.0f} ms after the end of the turn]\n")


async def _say(profile_dir: Path, voice_name: str | None) -> None:
    config = RuntimeConfig.load(profile_dir)
    recorder = start_session(config.observability, "say", viewer=False, console_level="WARNING")
    tts = config.tts if voice_name is None else config.tts.model_copy(update={"voice": voice_name})
    speaking = robot_speech(config.audio)
    async with open_voice_output(config.audio, tts, speaking, recorder) as voice:
        write("Type sentences to hear them. Ctrl+D to stop.\n")
        async for line in read_lines():
            started = time.perf_counter()
            spoken = await voice.say(split_sentences(_one(line)), lambda: True)
            write(f"[{(time.perf_counter() - started):.1f} s] {spoken.text}\n")


async def _run_voice_agent(
    profile_dir: Path,
    viewer: bool,
    camera: bool,
    show: bool,
    video: bool,
    ui: bool,
    depth: bool,
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
        monitor = _monitor(config, camera, show, stream=page is not None, on_key=None)
        recorder = _recorder(session, page, monitor)
        speaker = await _speaker(config.active_speaker) if camera else None
        vision = (
            await asyncio.to_thread(
                _face_vision, config.vision, recorder, video, depth=depth, speaker=speaker
            )
            if camera
            else None
        )
        # Without an active speaker, an utterance is the face in front's, said to the robot.
        delay_s = (
            None
            if speaker is None or config.active_speaker is None
            else config.active_speaker.delay_s
        )
        interaction = None if vision is None else Interaction(config.interaction, delay_s, recorder)
        async with (
            open_voice_output(config.audio, config.tts, speaking, recorder) as speech,
            open_voice_input(
                config.audio,
                config.asr,
                recorder,
                speaking,
                speech.interrupt,
                _anyone if interaction is None or delay_s is None else interaction.may_interrupt,
                speaker,
            ) as voice,
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
                monitor,
                interaction,
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
    monitor: CameraMonitor | None,
    interaction: Interaction | None,
) -> None:
    if interaction is not None:
        user_input = AttributedInput(user_input, interaction)
    agent = Agent(
        llm=OpenAiLanguageModel(config.llm),
        vla=vla,
        body=body,
        user_input=_captioned(user_input, vision),
        speech=speech,
        recorder=recorder,
        vision=vision,
        engagement=interaction,
        profile=profile,
        admins=load_admins(config.admins),
        config=config.agent,
    )
    logger.info("{} is up", profile.character.name)
    await _with_vision(agent.run(), vision, monitor, config.attention, interaction, recorder)


async def _one(text: str) -> AsyncGenerator[str]:
    yield text


@app.command
def see(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    viewer: bool = False,
    video: bool = False,
    stream: bool = False,
    window: bool = True,
    depth: bool = True,
) -> None:
    """Show the camera with each tracked face, who it is and what it shows, without the agent.

    Faces coming and going are logged. Press n in the window to type a name for the nearest face in
    the terminal, q to quit.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the per-frame timings, to a rerun viewer.
        video: Record the camera frames and their depth with the session, to replay them in rerun.
        stream: Also publish the window's image to MediaMTX, to watch on its WebRTC page.
        window: Show the debug window; without it, people are only logged, recorded and
            streamed.
        depth: Estimate the camera's depth, if the profile has a depth model.
    """
    _run(_see(profile, viewer, video, stream, window, depth, listen=False))


@app.command
def sense(
    *,
    profile: ProfileDir = _DEFAULT_PROFILE,
    viewer: bool = False,
    video: bool = False,
    window: bool = True,
    depth: bool = True,
) -> None:
    """Like `see`, with the microphone too: which face speaks, from its lips and the sound.

    Camera, microphone and where they meet, as in `run`, without the agent or any server.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the per-frame timings, to a rerun viewer.
        video: Record the camera frames and their depth with the session, to replay them in rerun.
        window: Show the debug window; without it, people are only logged and recorded.
        depth: Estimate the camera's depth, if the profile has a depth model.
    """
    _run(_see(profile, viewer, video, False, window, depth, listen=True))


async def _run_text_agent(
    profile_dir: Path,
    viewer: bool,
    camera: bool,
    show: bool,
    video: bool,
    ui: bool,
    depth: bool,
) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(*_agent_servers(config, streamed=ui and camera))
    session = start_session(config.observability, "text", viewer=viewer, console_level="WARNING")
    async with (
        _page(config, ui, camera) as page,
        HttpVlaClient(config.vla) as vla,
        HttpBodyClient(config.body) as body,
    ):
        monitor = _monitor(config, camera, show, stream=page is not None, on_key=None)
        recorder = _recorder(session, page, monitor)
        vision = (
            await asyncio.to_thread(
                _face_vision, config.vision, recorder, video, depth=depth, speaker=None
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
            monitor,
            # No microphone: typed lines are the face in front's, and people still engage.
            None if vision is None else Interaction(config.interaction, None, recorder),
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


async def _see(
    profile_dir: Path,
    viewer: bool,
    video: bool,
    stream: bool,
    window: bool,
    depth: bool,
    *,
    listen: bool,
) -> None:
    config = RuntimeConfig.load(profile_dir)
    if stream:
        await require_servers(_media_server(config))
    session = start_session(
        config.observability, "sense" if listen else "see", viewer=viewer, console_level="INFO"
    )
    keys = asyncio.Queue[str]()
    if listen and config.active_speaker is None:
        raise SystemExit(f"{profile_dir / 'runtime.yaml'} has no active_speaker to listen with")
    speaker = await _speaker(config.active_speaker) if listen else None
    closed = asyncio.Event()

    def on_key(key: str) -> None:
        if key == "q":
            closed.set()
        else:
            keys.put_nowait(key)

    monitor = _monitor(config, True, window, stream=stream, on_key=on_key)
    recorder = _recorder(session, None, monitor)
    interaction = Interaction(
        config.interaction,
        None if config.active_speaker is None or not listen else config.active_speaker.delay_s,
        recorder,
    )
    vision = await asyncio.to_thread(
        _face_vision, config.vision, recorder, video, depth=depth, speaker=speaker
    )
    if stream:
        logger.info("Streaming the window to {}", config.ui.video_page)
    try:
        async with contextlib.AsyncExitStack() as stack:
            group = await stack.enter_async_context(asyncio.TaskGroup())
            tasks = [
                group.create_task(_name_nearest_face(vision, keys)),
                group.create_task(_attend(vision, config.attention, recorder)),
                group.create_task(closed.wait()),
                # People engaging and disengaging are logged, and their scores recorded.
                group.create_task(_interact(vision, interaction)),
            ]
            if speaker is not None:
                microphone = await stack.enter_async_context(
                    Microphone(config.audio.input_device, speaker.hear)
                )
                # The speaker detector takes the sound in the audio thread; nothing else needs it.
                tasks.append(group.create_task(_drain(microphone.frames())))
            seeing = group.create_task(vision.run())
            # Until the camera fails or the window is closed.
            await asyncio.wait([seeing, tasks[2]], return_when=asyncio.FIRST_COMPLETED)
            for task in [*tasks, seeing]:
                task.cancel()
    finally:
        if monitor is not None:
            monitor.close()


def _monitor(
    config: RuntimeConfig,
    camera: bool,
    window: bool,
    *,
    stream: bool,
    on_key: Callable[[str], None] | None,
) -> CameraMonitor | None:
    """What shows the camera live, if anything does: the window, the stream, or both.

    `on_key` is called in the event loop with each key pressed in the window.
    """
    if not camera or not (window or stream):
        return None
    loop = asyncio.get_running_loop()

    def in_loop(key: str) -> None:
        assert on_key is not None
        loop.call_soon_threadsafe(on_key, key)

    return CameraMonitor(
        window=_WINDOW_SIZE if window else None,
        stream=config.stream if stream else None,
        on_key=None if on_key is None else in_loop,
    )


async def _attend(vision: FaceVision, config: VisualAttentionConfig, recorder: Recorder) -> None:
    """Attend to what the camera sees, until cancelled: nothing takes the target yet.

    Each state is recorded, so whatever shows the camera marks the target.
    """
    async with vision.observations() as observations:
        await _drain(follow(observations, VisualAttention(config), recorder))


async def _interact(vision: FaceVision, interaction: Interaction) -> None:
    """Show `interaction` what the camera sees in each frame, until cancelled."""
    async with vision.observations() as observations:
        await interaction.run(observations)


async def _speaker(config: ActiveSpeakerConfig | None) -> SpeakerDetector | None:
    return None if config is None else await asyncio.to_thread(open_light_asd, config)


async def _drain(items: AsyncIterator[object]) -> None:
    async for _ in items:
        pass


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
    depth: bool,
    speaker: SpeakerDetector | None,
) -> FaceVision:
    return FaceVision(
        config if depth else config.model_copy(update={"depth": None}),
        _face_database(config),
        recorder=recorder,
        record_video=record_video,
        speaker=speaker,
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
            match utterance.speaker:
                case None:
                    # Not attributed: taken to be the face in front, as the agent does.
                    nearest = next((face for face in self._vision.faces() if face.nearest), None)
                    if nearest is not None:
                        self._vision.caption(nearest.track_id, utterance.text)
                case SeenSpeaking(track_id=track_id):
                    self._vision.caption(track_id, utterance.text)
                case NobodySeenSpeaking():
                    pass
                case _:
                    assert_never(utterance.speaker)
            yield utterance


def _face_database(config: FaceVisionConfig) -> FaceDatabase:
    return FaceDatabase(config.database.expanduser(), config.max_embeddings_per_person)


async def _with_vision(
    main: Coroutine[None, None, None],
    vision: FaceVision | None,
    monitor: CameraMonitor | None,
    attention: VisualAttentionConfig,
    interaction: Interaction | None,
    recorder: Recorder,
) -> None:
    """Run `main` while the camera is watched, shown and attended to.

    With an `interaction`, it is shown what the camera sees meanwhile.

    A camera failure stops `main`; closing the window only closes it.
    """
    if vision is None:
        await main
        return
    try:
        async with asyncio.TaskGroup() as group:
            seeing = group.create_task(vision.run())
            watching = [group.create_task(_attend(vision, attention, recorder))]
            if interaction is not None:
                watching.append(group.create_task(_interact(vision, interaction)))
            await main
            seeing.cancel()
            for task in watching:
                task.cancel()
    finally:
        if monitor is not None:
            monitor.close()


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
