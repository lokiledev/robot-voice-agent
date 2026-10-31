import asyncio
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Final, assert_never
from uuid import UUID

from loguru import logger
from opentelemetry import trace

from rva_agent.body import BodySupervisor, BodyTools
from rva_agent.config import AgentConfig
from rva_agent.history import History
from rva_agent.mute import MuteTools
from rva_agent.people import PeopleTools
from rva_agent.permission import ActionPermission
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import split_sentences
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import TaskTools, ToolResult
from rva_body_schemas.abc.client import BodyClient
from rva_body_schemas.motion import GoToHuman, GoToLocation, Motion, PlayAnimation, describe
from rva_schemas.abc.conversation import SpeechOutput, UserInput
from rva_schemas.abc.llm import LanguageModel
from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.vision import Vision
from rva_schemas.dialogue import (
    AssistantMessage,
    RobotEventMessage,
    Spoken,
    TextDelta,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    UserUtterance,
)
from rva_schemas.profile import Profile
from rva_schemas.record import DialogueLine, FaceUpdate, MotionUpdate, TaskUpdate, ToolInvocation
from rva_schemas.vision import FaceAppeared, FaceEvent, FaceLeft
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.task import Task, TaskCancelled, TaskFailed, TaskRunning, TaskSucceeded

_tracer: Final = trace.get_tracer("rva_agent")
_PUNCTUATION: Final = re.compile(r"[^\w\s]")


class Agent:
    """The robot's dialogue loop: answers the user and drives the body through the VLA.

    With a `Vision`, it is also told when people come and go, and can remember their names.
    """

    def __init__(
        self,
        *,
        llm: LanguageModel,
        vla: VlaClient,
        body: BodyClient,
        user_input: UserInput,
        speech: SpeechOutput,
        recorder: Recorder,
        vision: Vision | None,
        profile: Profile,
        config: AgentConfig,
    ) -> None:
        self._llm: Final = llm
        self._vla: Final = vla
        self._body_client: Final = body
        self._user_input: Final = user_input
        self._speech: Final = speech
        self._recorder: Final = recorder
        self._vision: Final = vision
        self._config: Final = config
        # One turn at a time: user utterances, ended tasks and motions, and face events queue up.
        self._inbox: Final = asyncio.Queue[
            UserUtterance | Task | Motion | FaceAppeared | FaceLeft
        ]()
        self._supervisor: Final = TaskSupervisor(vla, self._inbox.put_nowait)
        self._body: Final = BodySupervisor(body, self._inbox.put_nowait)
        self._permission: Final = ActionPermission(vision, profile.admins)
        self._tools: Final = TaskTools(self._supervisor, self._moving, self._permission)
        self._body_tools: Final = BodyTools(
            self._body,
            self._supervisor,
            self._permission,
            vision,
            tuple(place.name for place in profile.environment.places),
            profile.capabilities.animations,
        )
        self._people: Final = None if vision is None else PeopleTools(vision, self._permission)
        self._mute: Final = MuteTools(self._permission)
        self._tool_specs: Final = (
            TaskTools.specs
            + MuteTools.specs
            + self._body_tools.specs
            + (() if self._people is None else PeopleTools.specs)
        )
        self._history: Final = History(
            render_system_prompt(profile, vision=vision is not None), config.max_history_messages
        )
        # When each known person was last seen leaving, by user id, on the monotonic clock.
        self._left_at: Final[dict[UUID, float]] = {}

    async def run(self) -> None:
        """Converse until the user input ends, then finish the turns already queued."""
        async with (
            self._vla.events() as events,
            self._body_client.events() as motions,
            self._face_events() as faces,
            asyncio.TaskGroup() as group,
        ):
            watching = group.create_task(self._supervisor.watch(events))
            moving = group.create_task(self._body.watch(motions))
            seeing = group.create_task(self._queue_faces(faces))
            handling = group.create_task(self._handle_inbox())
            async for utterance in self._user_input.utterances():
                self._inbox.put_nowait(utterance)
            await self._inbox.join()
            watching.cancel()
            moving.cancel()
            seeing.cancel()
            handling.cancel()

    @asynccontextmanager
    async def _face_events(self) -> AsyncGenerator[AsyncIterator[FaceEvent]]:
        if self._vision is None:
            yield _no_events()
            return
        async with self._vision.events() as events:
            yield events

    async def _queue_faces(self, events: AsyncIterator[FaceEvent]) -> None:
        async for event in events:
            self._inbox.put_nowait(event)

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
                        case Motion():
                            await self._on_motion_end(item)
                        case FaceAppeared() | FaceLeft():
                            await self._on_face(item)
                        case _:
                            assert_never(item)
            finally:
                self._inbox.task_done()

    async def _on_utterance(self, utterance: UserUtterance) -> None:
        self._recorder.record(DialogueLine(speaker="user", text=utterance.text))
        self._history.append(UserMessage(content=self._attributed(utterance.text)))
        busy = self._supervisor.active is not None or self._body.active is not None
        if self._is_stop_word(utterance.text) and busy:
            logger.info("Stop word: stopping the body without the LLM")
            await self._supervisor.cancel()
            await self._body.stay()
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

    async def _on_face(self, event: FaceAppeared | FaceLeft) -> None:
        self._recorder.record(FaceUpdate(event=event))
        self._history.append(RobotEventMessage(content=_describe_face(event, self._permission)))
        match event:
            case FaceAppeared():
                person = event.face.person
                left_at = None if person is None else self._left_at.get(person.user_id)
                if (
                    left_at is not None
                    and time.monotonic() - left_at < self._config.regreet_after_s
                ):
                    # Stepped out of view for a moment: already greeted.
                    return
                await self._respond()
            case FaceLeft():
                if event.person is not None:
                    self._left_at[event.person.user_id] = time.monotonic()
            case _:
                assert_never(event)

    async def _on_motion_end(self, motion: Motion) -> None:
        self._recorder.record(MotionUpdate(motion=motion))
        self._history.append(RobotEventMessage(content=_describe_motion_end(motion)))
        match motion.status, motion.action:
            # A finished gesture needs no comment; a stop was asked for.
            case TaskCancelled(), _:
                pass
            case TaskSucceeded(), PlayAnimation():
                pass
            case _:
                await self._respond()

    def _attributed(self, text: str) -> str:
        """`text` prefixed with who was in front of the robot when it was said.

        Without it, the LLM keeps answering whoever spoke last after someone else steps in front.
        """
        if self._mute.muted:
            # Said in the turn itself: the model follows it far more than a rule in its prompt.
            text = f"{text} (You are muted: answer with no words, only with tools if needed.)"
        if self._vision is None:
            return text
        nearest = next((face for face in self._vision.faces() if face.nearest), None)
        if nearest is None:
            return f"[nobody in view] {text}"
        if nearest.person is None:
            return f"[unknown person, track {nearest.track_id}] {text}"
        return f"[{nearest.person.name}, track {nearest.track_id}] {text}"

    def _moving(self) -> str | None:
        """What the body is doing besides a VLA task, if anything."""
        motion = self._body.active
        return None if motion is None else describe(motion.action)

    async def _respond(self) -> None:
        """Let the LLM speak and call tools until it has nothing left to do."""
        for _ in range(self._config.max_tool_rounds):
            text: list[str] = []
            tool_calls: list[ToolCall] = []
            # Decided per round: a mute or unmute takes effect from the next round of the turn.
            muted = self._mute.muted
            with _tracer.start_as_current_span("llm"):
                async with (
                    aclosing(self._stream_text(text, tool_calls)) as chunks,
                    aclosing(split_sentences(chunks)) as sentences,
                ):
                    spoken = await (_silently(sentences) if muted else self._speech.say(sentences))
            if muted:
                # Nothing was heard, so nothing was said.
                text.clear()
            if spoken.interrupted:
                # The user talked over the robot: keep what they heard, drop the rest and any action.
                self._add_robot_message(AssistantMessage(content=spoken.text, tool_calls=()))
                return
            self._add_robot_message(
                AssistantMessage(content="".join(text), tool_calls=tuple(tool_calls))
            )
            if not tool_calls:
                return
            needs_reply = False
            for call in tool_calls:
                with _tracer.start_as_current_span(f"tool:{call.name}"):
                    result = await self._call_tool(call)
                self._recorder.record(
                    ToolInvocation(name=call.name, arguments=call.arguments, result=result.content)
                )
                self._history.append(
                    ToolResultMessage(tool_call_id=call.id, content=result.content)
                )
                needs_reply = needs_reply or result.needs_reply
            # Asking the LLM again here would only make it repeat what it just announced, or say
            # nothing while muted.
            if (text or muted) and not needs_reply:
                return
        logger.warning(
            "The LLM was still calling tools after {} rounds", self._config.max_tool_rounds
        )

    async def _call_tool(self, call: ToolCall) -> ToolResult:
        if call.name in MuteTools.names:
            return await self._mute.call(call)
        if self._people is not None and call.name in PeopleTools.names:
            return await self._people.call(call)
        if call.name in BodyTools.names:
            return await self._body_tools.call(call)
        return await self._tools.call(call)

    async def _stream_text(
        self, text: list[str], tool_calls: list[ToolCall]
    ) -> AsyncGenerator[str]:
        span = trace.get_current_span()
        async for output in self._llm.stream(self._history.messages, self._tool_specs):
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
        if self._mute.muted:
            return
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
            return f"Your body finished: {task.instruction}. It worked."
        case TaskFailed(cause="timeout"):
            return f"Your body gave up, it took too long: {task.instruction}. {task.status.reason}."
        case TaskFailed():
            return f"Your body could not finish: {task.instruction}. Reason: {task.status.reason}."
        case TaskCancelled():
            return f"Your body stopped before finishing: {task.instruction}."
        case TaskRunning():
            raise AssertionError(f"Task {task.id} is still running")
        case _:
            assert_never(task.status)


def _describe_motion_end(motion: Motion) -> str:
    action = motion.action
    match motion.status:
        case TaskSucceeded():
            match action:
                case GoToLocation():
                    return f"You arrived at the {action.location}."
                case GoToHuman():
                    return "You reached the person you were walking to."
                case PlayAnimation():
                    return f"You finished the {action.animation} animation."
                case _:
                    assert_never(action)
        case TaskFailed(cause="timeout"):
            return f"Your body gave up, it took too long: {describe(action)}."
        case TaskFailed():
            return f"Your body could not {describe(action)}. Reason: {motion.status.reason}."
        case TaskCancelled():
            return f"You stopped before you could {describe(action)}."
        case TaskRunning():
            raise AssertionError(f"Motion {motion.id} reported as ended while running")
        case _:
            assert_never(motion.status)


def _describe_face(event: FaceAppeared | FaceLeft, permission: ActionPermission) -> str:
    match event:
        case FaceAppeared():
            face = event.face
            where = f"track {face.track_id}, nearest" if face.nearest else f"track {face.track_id}"
            # Said in the event itself: the model follows it far more than a rule in its prompt.
            greet = " Greet them: look at them and wave." if face.nearest else ""
            if face.person is None:
                return f"An unknown person is in view ({where}).{greet}"
            # Said up front so the LLM declines a non-operator's request instead of trying it.
            role = "one of" if permission.is_operator(face.person.user_id) else "not one of"
            return (
                f"{face.person.name} is in view ({where}, user {face.person.user_id},"
                f" {role} your operators).{greet}"
            )
        case FaceLeft():
            if event.person is None:
                return f"The unknown person on track {event.track_id} left."
            return f"{event.person.name} left (track {event.track_id})."
        case _:
            assert_never(event)


async def _silently(sentences: AsyncIterator[str]) -> Spoken:
    """Consume the LLM's sentences without speaking them."""
    async for _ in sentences:
        pass
    return Spoken(text="", interrupted=False)


async def _single(text: str) -> AsyncIterator[str]:
    yield text


async def _no_events() -> AsyncIterator[FaceEvent]:
    return
    yield
