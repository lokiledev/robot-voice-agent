import asyncio
import os
import sys
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, assert_never

import cyclopts
from loguru import logger

from rva_agent.agent import Agent
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import split_sentences
from rva_agent.tools import TaskTools
from rva_app.config import RuntimeConfig
from rva_app.servers import Server, ServersUnavailableError, require_servers
from rva_app.terminal import TerminalSpeechOutput, TerminalUserInput, read_lines, write
from rva_app.voice import open_voice_input, open_voice_output, robot_speech
from rva_llm.client import OpenAiLanguageModel
from rva_llm.server import llama_server_command
from rva_observability.session import start_session
from rva_schemas.dialogue import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextDelta,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from rva_schemas.profile import Profile
from rva_schemas.vision import UnknownTrackError
from rva_vision_faces.config import FaceVisionConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.vision import FaceVision
from rva_vla_client.client import HttpVlaClient

app = cyclopts.App(name="rva", help="Robot voice agent and tools to try each part on its own.")

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
) -> None:
    """Run the agent in the terminal: type to talk, read its replies. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
    """
    _run(_run_text_agent(profile, viewer, camera, show))


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
) -> None:
    """Run the voice agent. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
        camera: Watch for people with the camera.
        show: Also show the camera with the tracked faces in a window.
    """
    _run(_run_voice_agent(profile, viewer, camera, show))


def _run(main: Coroutine[object, object, None]) -> None:
    try:
        asyncio.run(main)
    except ServersUnavailableError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error


def _llm_server(config: RuntimeConfig) -> Server:
    return Server(
        name="LLM server",
        url=f"{config.llm.base_url}",
        how_to_start="start it with `uv run rva serve-llm`",
    )


def _vla_server(config: RuntimeConfig) -> Server:
    return Server(
        name="VLA",
        url=f"{config.vla.base_url}",
        how_to_start="start the mock with `uv run rva-vla-mock --interactive`, or set vla.base_url in runtime.yaml",
    )


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
    tts = config.tts if voice_name is None else config.tts.model_copy(update={"voice": voice_name})
    async with open_voice_output(config.audio, tts, robot_speech(config.audio)) as voice:
        write("Type sentences to hear them. Ctrl+D to stop.\n")
        async for line in read_lines():
            started = time.perf_counter()
            spoken = await voice.say(split_sentences(_one(line)))
            write(f"[{(time.perf_counter() - started):.1f} s] {spoken.text}\n")


async def _run_voice_agent(profile_dir: Path, viewer: bool, camera: bool, show: bool) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config), _vla_server(config))
    recorder = start_session(config.observability, "voice", viewer=viewer, console_level="INFO")
    vision = (
        await asyncio.to_thread(_face_vision, config.vision, show_window=show, on_key=None)
        if camera
        else None
    )
    speaking = robot_speech(config.audio)
    async with (
        HttpVlaClient(config.vla) as vla,
        open_voice_output(config.audio, config.tts, speaking) as speech,
        open_voice_input(config.audio, recorder, speaking, speech.interrupt) as user_input,
    ):
        agent = Agent(
            llm=OpenAiLanguageModel(config.llm),
            vla=vla,
            user_input=user_input,
            speech=speech,
            recorder=recorder,
            vision=vision,
            profile=profile,
            config=config.agent,
        )
        logger.info("{} is listening", profile.character.name)
        await _with_vision(agent.run(), vision)


async def _one(text: str) -> AsyncGenerator[str]:
    yield text


@app.command
def see(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Show the camera with each tracked face and who it is, without the agent.

    Faces coming and going are logged. Press n in the window to type a name for the nearest face in
    the terminal, q to quit.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the per-frame timings, to a rerun viewer.
    """
    asyncio.run(_see(profile, viewer))


async def _run_text_agent(profile_dir: Path, viewer: bool, camera: bool, show: bool) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config), _vla_server(config))
    recorder = start_session(config.observability, "text", viewer=viewer, console_level="WARNING")
    vision = (
        await asyncio.to_thread(_face_vision, config.vision, show_window=show, on_key=None)
        if camera
        else None
    )
    async with HttpVlaClient(config.vla) as vla:
        agent = Agent(
            llm=OpenAiLanguageModel(config.llm),
            vla=vla,
            user_input=TerminalUserInput(),
            speech=TerminalSpeechOutput(profile.character.name),
            recorder=recorder,
            vision=vision,
            profile=profile,
            config=config.agent,
        )
        await _with_vision(agent.run(), vision)


@app.command
def people(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """List the user id and name of everyone the robot knows, and whether they are an operator."""
    config = RuntimeConfig.load(profile)
    operators = Profile.load(profile).operators
    known = _face_database(config.vision).people()
    if not known:
        write("Nobody is known yet. Name a face with `rva see`, then n.\n")
    for person in known:
        role = "  operator" if person.user_id in operators.user_ids else ""
        write(f"{person.user_id}  {person.name}{role}\n")


async def _see(profile_dir: Path, viewer: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    start_session(config.observability, "see", viewer=viewer, console_level="INFO")
    keys = asyncio.Queue[str]()
    vision = await asyncio.to_thread(
        _face_vision, config.vision, show_window=True, on_key=keys.put_nowait
    )
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
    config: FaceVisionConfig, *, show_window: bool, on_key: Callable[[str], None] | None
) -> FaceVision:
    return FaceVision(config, _face_database(config), show_window=show_window, on_key=on_key)


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
            ToolResultMessage(tool_call_id=call.id, content='{"status": "not run in rva llm"}')
            for call in calls
        )
