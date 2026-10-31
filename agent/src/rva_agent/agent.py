import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Final, assert_never

from loguru import logger
from opentelemetry import trace

from rva_agent.config import AgentConfig
from rva_agent.history import History
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import split_sentences
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import TaskTools
from rva_schemas.abc.conversation import SpeechOutput, UserInput
from rva_schemas.abc.llm import LanguageModel
from rva_schemas.abc.recorder import Recorder
from rva_schemas.dialogue import (
    AssistantMessage,
    RobotEventMessage,
    TextDelta,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    UserUtterance,
)
from rva_schemas.profile import Profile
from rva_schemas.record import DialogueLine, TaskUpdate, ToolInvocation
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.task import Task, TaskCancelled, TaskFailed, TaskRunning, TaskSucceeded

_tracer: Final = trace.get_tracer("rva_agent")
_PUNCTUATION: Final = re.compile(r"[^\w\s]")


class Agent:
    """The robot's dialogue loop: answers the user and drives the body through the VLA."""

    def __init__(
        self,
        *,
        llm: LanguageModel,
        vla: VlaClient,
        user_input: UserInput,
        speech: SpeechOutput,
        recorder: Recorder,
        profile: Profile,
        config: AgentConfig,
    ) -> None:
        self._llm: Final = llm
        self._vla: Final = vla
        self._user_input: Final = user_input
        self._speech: Final = speech
        self._recorder: Final = recorder
        self._config: Final = config
        # One turn at a time: user utterances and ended tasks queue up here.
        self._inbox: Final = asyncio.Queue[UserUtterance | Task]()
        self._supervisor: Final = TaskSupervisor(vla, self._inbox.put_nowait)
        self._tools: Final = TaskTools(self._supervisor)
        self._history: Final = History(render_system_prompt(profile), config.max_history_messages)

    async def run(self) -> None:
        """Converse until the user input ends, then finish the turns already queued."""
        async with self._vla.events() as events, asyncio.TaskGroup() as group:
            watching = group.create_task(self._supervisor.watch(events))
            handling = group.create_task(self._handle_inbox())
            async for utterance in self._user_input.utterances():
                self._inbox.put_nowait(utterance)
            await self._inbox.join()
            watching.cancel()
            handling.cancel()

    async def _handle_inbox(self) -> None:
        while True:
            item = await self._inbox.get()
            try:
                with _tracer.start_as_current_span("turn"):
                    match item:
                        case UserUtterance():
                            await self._on_utterance(item)
                        case Task():
                            await self._on_task_end(item)
                        case _:
                            assert_never(item)
            finally:
                self._inbox.task_done()

    async def _on_utterance(self, utterance: UserUtterance) -> None:
        self._recorder.record(DialogueLine(speaker="user", text=utterance.text))
        self._history.append(UserMessage(content=utterance.text))
        if self._is_stop_word(utterance.text) and self._supervisor.active is not None:
            logger.info("Stop word: cancelling the task without the LLM")
            await self._supervisor.cancel()
            await self._say_fixed("Stopping.")
            return
        await self._respond()

    async def _on_task_end(self, task: Task) -> None:
        self._recorder.record(TaskUpdate(task=task))
        self._history.append(RobotEventMessage(content=_describe_end(task)))
        match task.status:
            case TaskCancelled():
                # Someone asked for it, the user already knows.
                pass
            case TaskSucceeded() | TaskFailed():
                await self._respond()
            case TaskRunning():
                raise AssertionError(f"Task {task.id} reported as ended while running")
            case _:
                assert_never(task.status)

    async def _respond(self) -> None:
        """Let the LLM speak and call tools until it has nothing left to do."""
        for _ in range(self._config.max_tool_rounds):
            text: list[str] = []
            tool_calls: list[ToolCall] = []
            with _tracer.start_as_current_span("llm"):
                async with (
                    aclosing(self._stream_text(text, tool_calls)) as chunks,
                    aclosing(split_sentences(chunks)) as sentences,
                ):
                    spoken = await self._speech.say(sentences)
            if spoken.interrupted:
                # The user talked over the robot: keep what they heard, drop the rest and any action.
                self._add_robot_message(AssistantMessage(content=spoken.text, tool_calls=()))
                return
            self._add_robot_message(
                AssistantMessage(content="".join(text), tool_calls=tuple(tool_calls))
            )
            if not tool_calls:
                return
            for call in tool_calls:
                with _tracer.start_as_current_span(f"tool:{call.name}"):
                    result = await self._tools.call(call)
                self._recorder.record(
                    ToolInvocation(name=call.name, arguments=call.arguments, result=result)
                )
                self._history.append(ToolResultMessage(tool_call_id=call.id, content=result))
        logger.warning(
            "The LLM was still calling tools after {} rounds", self._config.max_tool_rounds
        )

    async def _stream_text(
        self, text: list[str], tool_calls: list[ToolCall]
    ) -> AsyncGenerator[str]:
        span = trace.get_current_span()
        async for output in self._llm.stream(self._history.messages, self._tools.specs):
            match output:
                case TextDelta():
                    if not text:
                        span.add_event("first_text")
                    text.append(output.text)
                    yield output.text
                case ToolCall():
                    tool_calls.append(output)
                case _:
                    assert_never(output)

    async def _say_fixed(self, text: str) -> None:
        spoken = await self._speech.say(_single(text))
        self._add_robot_message(AssistantMessage(content=spoken.text, tool_calls=()))

    def _add_robot_message(self, message: AssistantMessage) -> None:
        self._history.append(message)
        if message.content.strip():
            self._recorder.record(DialogueLine(speaker="robot", text=message.content))

    def _is_stop_word(self, text: str) -> bool:
        return _PUNCTUATION.sub("", text).strip().lower() in self._config.stop_words


def _describe_end(task: Task) -> str:
    match task.status:
        case TaskSucceeded():
            return f"The task {task.instruction!r} succeeded."
        case TaskFailed():
            return f"The task {task.instruction!r} failed: {task.status.reason}."
        case TaskCancelled():
            return f"The task {task.instruction!r} was cancelled."
        case TaskRunning():
            raise AssertionError(f"Task {task.id} is still running")
        case _:
            assert_never(task.status)


async def _single(text: str) -> AsyncIterator[str]:
    yield text
