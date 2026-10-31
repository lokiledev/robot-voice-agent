import asyncio
import os
from pathlib import Path
from typing import Annotated, assert_never

import cyclopts

from rva_agent.agent import Agent
from rva_agent.prompt import render_system_prompt
from rva_agent.tools import TaskTools
from rva_app.config import RuntimeConfig
from rva_app.terminal import TerminalSpeechOutput, TerminalUserInput, read_lines, write
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
    asyncio.run(_chat_with_llm(profile, system_prompt))


@app.command
def text(*, profile: ProfileDir = _DEFAULT_PROFILE, viewer: bool = False) -> None:
    """Run the agent in the terminal: type to talk, read its replies. Needs the LLM and VLA servers.

    Args:
        profile: Robot profile directory.
        viewer: Stream the recording to a rerun viewer.
    """
    asyncio.run(_run_text_agent(profile, viewer))


async def _run_text_agent(profile_dir: Path, viewer: bool) -> None:
    profile, config = Profile.load(profile_dir), RuntimeConfig.load(profile_dir)
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
