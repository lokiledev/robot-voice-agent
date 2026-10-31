import asyncio
import os
import sys
import time
from collections.abc import AsyncGenerator, Coroutine
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
from rva_vla_client.client import HttpVlaClient

app = cyclopts.App(name="rva", help="Robot voice agent and tools to try each part on its own.")

type ProfileDir = Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")]
_DEFAULT_PROFILE = Path("config/default")


@app.command
def prompt(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """Print the system prompt rendered from a profile."""
    write(render_system_prompt(Profile.load(profile)))


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
def text(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Run the agent in the terminal: type to talk, read its replies. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
    """
    _run(_run_text_agent(profile, viewer))


@app.command
def listen(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Print what the microphone hears, turn by turn: VAD, end of turn and ASR, without the agent.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording, with the VAD probability, to a rerun viewer.
    """
    asyncio.run(_listen(profile, viewer))


@app.command
def say(*, profile: ProfileDir = _DEFAULT_PROFILE) -> None:
    """Speak each line typed, sentence by sentence, with the profile's voice."""
    asyncio.run(_say(profile))


@app.command
def run(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Run the voice agent. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
    """
    _run(_run_voice_agent(profile, viewer))


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


async def _say(profile_dir: Path) -> None:
    config = RuntimeConfig.load(profile_dir)
    async with open_voice_output(config.audio, config.tts, robot_speech(config.audio)) as voice:
        write("Type sentences to hear them. Ctrl+D to stop.\n")
        async for line in read_lines():
            started = time.perf_counter()
            spoken = await voice.say(split_sentences(_one(line)))
            write(f"[{(time.perf_counter() - started):.1f} s] {spoken.text}\n")


async def _run_voice_agent(profile_dir: Path, viewer: bool) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config), _vla_server(config))
    recorder = start_session(config.observability, "voice", viewer=viewer, console_level="INFO")
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
            profile=profile,
            config=config.agent,
        )
        logger.info("{} is listening", profile.character.name)
        await agent.run()


async def _one(text: str) -> AsyncGenerator[str]:
    yield text


async def _run_text_agent(profile_dir: Path, viewer: bool) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config), _vla_server(config))
    recorder = start_session(config.observability, "text", viewer=viewer, console_level="WARNING")
    async with HttpVlaClient(config.vla) as vla:
        agent = Agent(
            llm=OpenAiLanguageModel(config.llm),
            vla=vla,
            user_input=TerminalUserInput(),
            speech=TerminalSpeechOutput(profile.character.name),
            recorder=recorder,
            profile=profile,
            config=config.agent,
        )
        await agent.run()


async def _chat_with_llm(profile_dir: Path, system_prompt: bool) -> None:
    config = RuntimeConfig.load(profile_dir)
    await require_servers(_llm_server(config))
    model = OpenAiLanguageModel(config.llm)
    tools = TaskTools.specs if system_prompt else ()
    messages: list[Message] = []
    if system_prompt:
        messages.append(SystemMessage(content=render_system_prompt(Profile.load(profile_dir))))
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
