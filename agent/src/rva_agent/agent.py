import asyncio
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from typing import Final, Never, assert_never
from uuid import UUID

from loguru import logger
from opentelemetry import trace

from rva_agent.body import BodySupervisor, BodyTools
from rva_agent.config import AgentConfig
from rva_agent.history import History
from rva_agent.people import PeopleTools
from rva_agent.permission import ActionPermission
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import split_sentences
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import TaskTools, ToolResult
from rva_body_schemas.abc.client import BodyClient
from rva_body_schemas.motion import GoToHuman, GoToLocation, Motion, PlayAnimation, describe
from rva_schemas.abc.conversation import SpeechOutput, UserInput
from rva_schemas.abc.engagement import Engagement
from rva_schemas.abc.llm import LanguageModel
from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.vision import Vision
from rva_schemas.dialogue import (
    AssistantMessage,
    NobodySeenSpeaking,
    RobotEventMessage,
    SeenSpeaking,
    Spoken,
    TextDelta,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    UserUtterance,
    UtteranceSpeaker,
)
from rva_schemas.interaction import Disengaged, Engaged, EngagementEvent
from rva_schemas.profile import Admins, Profile
from rva_schemas.record import (
    AgentStarted,
    FaceUpdate,
    MuteUpdate,
    RobotLine,
    Speaker,
    ToolInvocation,
    UserLine,
)
from rva_schemas.vision import FaceAppeared, FaceEvent, FaceLeft, KnownPerson
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.task import Task, TaskCancelled, TaskFailed, TaskRunning, TaskSucceeded

_tracer: Final = trace.get_tracer("rva_agent")
_PUNCTUATION: Final = re.compile(r"[^\w\s]")
# Told with each failure: deep in a long conversation, the LLM otherwise reports some of them from
# the outside ("the task to gather the cables failed") despite its instructions.
_OWN_FAILURE: Final = ' Say it as your own failure, like "I couldn\'t ..."; never call it a task.'
_REPEATED: Final = ToolResult(
    content='{"error": "You just called this, and it was not run again. Do not call it again now."}',
    needs_reply=False,
)


class Agent:
    """The robot's dialogue loop: answers the user and drives the body through the VLA.

    With a `Vision`, it can remember people's names. With an `Engagement`, it greets whoever
    starts paying attention to it, and stops talking to someone who no longer listens.
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
        engagement: Engagement | None,
        profile: Profile,
        admins: Admins,
        config: AgentConfig,
    ) -> None:
        self._llm: Final = llm
        self._vla: Final = vla
        self._body_client: Final = body
        self._user_input: Final = user_input
        self._speech: Final = speech
        self._recorder: Final = recorder
        self._vision: Final = vision
        self._engagement: Final = engagement
        self._config: Final = config
        self._name: Final = profile.character.name
        # One turn at a time: user utterances, ended tasks and motions, and people coming, going
        # and engaging queue up.
        self._inbox: Final = asyncio.Queue[
            UserUtterance | Task | Motion | FaceAppeared | FaceLeft | Engaged | Disengaged
        ]()
        self._supervisor: Final = TaskSupervisor(vla, recorder, self._inbox.put_nowait)
        self._body: Final = BodySupervisor(body, recorder, self._inbox.put_nowait)
        self._permission: Final = ActionPermission(vision, admins)
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
        self._muted = False
        self._tool_specs: Final = (
            TaskTools.specs
            + self._body_tools.specs
            + (() if self._people is None else PeopleTools.specs)
        )
        self._history: Final = History(
            render_system_prompt(profile, vision=vision is not None), config.max_history_messages
        )
        # When each known person last disengaged, by user id, on the monotonic clock.
        self._disengaged_at: Final[dict[UUID, float]] = {}
        # The tracks already greeted or heard from: a face is greeted once.
        self._greeted: Final[set[int]] = set()
        # The track of whoever stopped listening in the middle of a reply, until they are back.
        self._cut_short_for: int | None = None

    async def run(self) -> None:
        """Converse until the user input ends, then finish the turns already queued."""
        self._recorder.record(AgentStarted(name=self._name, tools=self._tool_specs))
        async with (
            self._vla.events() as events,
            self._body_client.events() as motions,
            self._face_events() as faces,
            self._engagement_events() as engagements,
            asyncio.TaskGroup() as group,
        ):
            background = [
                group.create_task(self._supervisor.watch(events)),
                group.create_task(self._body.watch(motions)),
                group.create_task(self._queue(faces)),
                group.create_task(self._queue(engagements)),
                group.create_task(self._handle_inbox()),
            ]
            async for utterance in self._user_input.utterances():
                if self._is_stop_word(utterance.text):
                    # Before its turn comes: a reply being said must not delay a stop.
                    self._speech.interrupt()
                self._inbox.put_nowait(utterance)
            await self._inbox.join()
            for task in background:
                task.cancel()

    @asynccontextmanager
    async def _face_events(self) -> AsyncGenerator[AsyncIterator[FaceEvent]]:
        if self._vision is None:
            yield _no_events()
            return
        async with self._vision.events() as events:
            yield events

    @asynccontextmanager
    async def _engagement_events(self) -> AsyncGenerator[AsyncIterator[EngagementEvent]]:
        if self._engagement is None:
            yield _no_events()
            return
        async with self._engagement.events() as events:
            yield events

    async def _queue(
        self, events: AsyncIterator[FaceEvent] | AsyncIterator[EngagementEvent]
    ) -> None:
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
                            self._on_face(item)
                        case Engaged():
                            await self._on_engaged(item)
                        case Disengaged():
                            self._on_disengaged(item)
                        case _:
                            assert_never(item)
            finally:
                self._inbox.task_done()

    async def _on_utterance(self, utterance: UserUtterance) -> None:
        speaker = self._speaker_of(utterance)
        busy = self._supervisor.active is not None or self._body.active is not None
        stopping = self._is_stop_word(utterance.text) and busy
        if not _said_to_robot(speaker) and not stopping:
            # Anyone may stop the body; the rest of what is not said to the robot is not heard.
            logger.info("Not said to the robot, dropped: {!r} from {}", utterance.text, speaker)
            return
        listener = speaker.track_id if isinstance(speaker, SeenSpeaking) else None
        if listener is not None:
            self._greeted.add(listener)
        self._recorder.record(
            UserLine(text=utterance.text, speaker=_speaker(speaker, self._permission))
        )
        self._history.append(UserMessage(content=self._attributed(utterance.text, speaker)))
        # Muting is matched in code rather than left to the LLM: it must work every time.
        if self._muted and _mentions(utterance.text, self._config.unmute_phrases):
            await self._unmute()
            return
        if not self._muted and _mentions(utterance.text, self._config.mute_phrases):
            await self._mute()
            return
        if stopping:
            logger.info("Stop word: stopping the body without the LLM")
            await self._supervisor.cancel()
            await self._body.stay()
            await self._say_fixed("Stopping.")
            return
        await self._respond(listener)

    async def _on_task_end(self, task: Task) -> None:
        self._history.append(RobotEventMessage(content=_describe_end(task)))
        match task.status:
            case TaskCancelled():
                # Someone asked for it, the user already knows.
                pass
            case TaskSucceeded() | TaskFailed():
                await self._respond(None)
            case TaskRunning():
                raise AssertionError(f"Task {task.id} reported as ended while running")
            case _:
                assert_never(task.status)

    def _on_face(self, event: FaceAppeared | FaceLeft) -> None:
        # Only recorded: the LLM hears of people once they engage, not of everyone passing by.
        self._recorder.record(FaceUpdate(event=event))
        if isinstance(event, FaceLeft):
            self._greeted.discard(event.track_id)

    async def _on_engaged(self, event: Engaged) -> None:
        who = _describe_person(event.track_id, event.person, self._permission)
        if self._cut_short_for == event.track_id:
            self._cut_short_for = None
            self._history.append(
                RobotEventMessage(
                    content=f"{who} is listening to you again. You had stopped in the middle of"
                    " your answer: pick it up if it still matters, briefly."
                )
            )
            await self._respond(event.track_id)
            return
        if event.track_id in self._greeted or self._back_soon(event.person):
            self._greeted.add(event.track_id)
            self._history.append(
                RobotEventMessage(content=f"{who} is paying attention to you again.")
            )
            return
        self._greeted.add(event.track_id)
        # Said in the event itself: the model follows it far more than a rule in its prompt.
        self._history.append(
            RobotEventMessage(
                content=f"{who} came to you and is looking at you. Greet them: look at them and"
                " wave."
            )
        )
        await self._respond(event.track_id)

    def _on_disengaged(self, event: Disengaged) -> None:
        if event.person is not None:
            self._disengaged_at[event.person.user_id] = time.monotonic()
        who = _name(event.track_id, event.person)
        what = "left" if event.left else "is no longer paying attention to you"
        self._history.append(RobotEventMessage(content=f"{who} {what}."))

    def _back_soon(self, person: KnownPerson | None) -> bool:
        """Whether a known person is back soon enough after disengaging not to be greeted again."""
        at = None if person is None else self._disengaged_at.get(person.user_id)
        return at is not None and time.monotonic() - at < self._config.regreet_after_s

    async def _on_motion_end(self, motion: Motion) -> None:
        self._history.append(RobotEventMessage(content=_describe_motion_end(motion)))
        match motion.status, motion.action:
            # A finished gesture needs no comment; a stop was asked for.
            case TaskCancelled(), _:
                pass
            case TaskSucceeded(), PlayAnimation():
                pass
            case _:
                await self._respond(None)

    def _speaker_of(self, utterance: UserUtterance) -> UtteranceSpeaker | None:
        """Who said `utterance`; `None` with nobody in view to have said it, or no camera.

        The face seen speaking it, where the camera and the microphone were put together.
        Otherwise, and for a typed line, the face in front of the robot now.
        """
        if utterance.speaker is not None:
            return utterance.speaker
        if self._vision is None:
            return None
        nearest = next((face for face in self._vision.faces() if face.nearest), None)
        if nearest is None:
            return None
        return SeenSpeaking(track_id=nearest.track_id, person=nearest.person, to_robot=True)

    def _attributed(self, text: str, speaker: UtteranceSpeaker | None) -> str:
        """`text` prefixed with who said it and with what the body is busy with, if anything.

        Without these, the LLM keeps answering whoever spoke last after someone else speaks, and
        promises an action before finding out the body is busy.
        """
        task = self._supervisor.active
        busy = task.instruction if task is not None else self._moving()
        if busy is not None:
            text = f"(Your body is busy: {busy}.) {text}"
        if self._muted:
            # Said in the turn itself: the model follows it far more than a rule in its prompt.
            text = f"{text} (You are muted: answer with no words, only with tools if needed.)"
        if self._vision is None:
            return text
        match speaker:
            case None:
                return f"[nobody in view] {text}"
            case NobodySeenSpeaking():
                return f"[someone out of view] {text}"
            case SeenSpeaking(person=person):
                who = "unknown person" if person is None else person.name
                return f"[{who}, track {speaker.track_id}] {text}"
            case _:
                assert_never(speaker)

    def _moving(self) -> str | None:
        """What the body is doing besides a VLA task, if anything."""
        motion = self._body.active
        return None if motion is None else describe(motion.action)

    async def _respond(self, listener: int | None) -> None:
        """Let the LLM speak and call tools until it has nothing left to do.

        `listener` is the track of the person answered, if it is one person: when they are engaged
        as the reply starts and stop being so, the reply stops after its sentence.
        """
        keep_going = self._while_listening(listener)
        # The calls made in this turn, by name and arguments.
        called: set[tuple[str, str]] = set()
        for _ in range(self._config.max_tool_rounds):
            text: list[str] = []
            tool_calls: list[ToolCall] = []
            # Decided per round: a mute or unmute takes effect from the next round of the turn.
            muted = self._muted
            with _tracer.start_as_current_span("llm"):
                async with (
                    aclosing(self._stream_text(text, tool_calls)) as chunks,
                    aclosing(split_sentences(chunks)) as sentences,
                ):
                    spoken = await (
                        _silently(sentences) if muted else self._speech.say(sentences, keep_going)
                    )
            if muted:
                # Nothing was heard, so nothing was said.
                text.clear()
            match spoken.end:
                case "talked_over":
                    # Keep what the user heard, drop the rest and any action.
                    self._add_robot_message(
                        AssistantMessage(content=spoken.text, tool_calls=()), interrupted=True
                    )
                    return
                case "cut_short":
                    self._add_robot_message(
                        AssistantMessage(content=spoken.text, tool_calls=()), interrupted=True
                    )
                    self._history.append(
                        RobotEventMessage(
                            content="You stopped talking there: they were no longer listening."
                        )
                    )
                    self._cut_short_for = listener
                    return
                case "finished":
                    pass
                case _:
                    assert_never(spoken.end)
            self._add_robot_message(
                AssistantMessage(content="".join(text), tool_calls=tuple(tool_calls)),
                interrupted=False,
            )
            if not tool_calls:
                return
            needs_reply = False
            for call in tool_calls:
                if (call.name, call.arguments) in called:
                    # The LLM retries what just failed, round after round, saying each time that
                    # it does it now: the turn ends here instead.
                    result = _REPEATED
                else:
                    called.add((call.name, call.arguments))
                    with _tracer.start_as_current_span(f"tool:{call.name}"):
                        result = await self._call_tool(call)
                self._recorder.record(
                    ToolInvocation(name=call.name, arguments=call.arguments, result=result.content)
                )
                self._history.append(
                    ToolResultMessage(tool_call_id=call.id, content=result.content)
                )
                needs_reply = needs_reply or result.needs_reply
            # Asking the LLM again here would only make it repeat what it just announced, or, with
            # nothing left to say, make something up: an event it expects, said aloud.
            if not needs_reply:
                return
        logger.warning(
            "The LLM was still calling tools after {} rounds", self._config.max_tool_rounds
        )

    def _while_listening(self, listener: int | None) -> Callable[[], bool]:
        """Whether to keep talking: until a `listener` engaged now no longer is."""
        engagement = self._engagement
        if engagement is None or listener is None or listener not in engagement.engaged():
            return _always
        return lambda: listener in engagement.engaged()

    async def _call_tool(self, call: ToolCall) -> ToolResult:
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
        if self._muted:
            return
        spoken = await self._speech.say(_single(text), _always)
        self._add_robot_message(
            AssistantMessage(content=spoken.text, tool_calls=()),
            interrupted=spoken.end != "finished",
        )

    def _add_robot_message(self, message: AssistantMessage, *, interrupted: bool) -> None:
        self._history.append(message)
        # Interrupted in its first sentence, a reply has no words heard, but one was started.
        if message.content.strip() or interrupted:
            self._recorder.record(RobotLine(text=message.content, interrupted=interrupted))

    def _is_stop_word(self, text: str) -> bool:
        return _normalized(text) in self._config.stop_words

    async def _mute(self) -> None:
        await self._say_fixed("Okay.")
        self._muted = True
        self._recorder.record(MuteUpdate(muted=True))
        logger.info("Muted")
        self._history.append(
            RobotEventMessage(content="You are muted: say nothing until an operator lets you talk.")
        )

    async def _unmute(self) -> None:
        if (refusal := self._permission.refusal()) is not None:
            logger.info("Not unmuted: {}", refusal)
            self._history.append(
                RobotEventMessage(
                    content=f"Someone asked you to talk again. {refusal} Only an operator can let"
                    " you talk again, so you stay quiet."
                )
            )
            return
        self._muted = False
        self._recorder.record(MuteUpdate(muted=False))
        logger.info("Unmuted")
        self._history.append(RobotEventMessage(content="An operator let you talk again."))
        await self._say_fixed("I can talk again.")


def _describe_end(task: Task) -> str:
    match task.status:
        case TaskSucceeded():
            return f"You managed to {task.instruction}."
        case TaskFailed(cause="timeout"):
            return (
                f"You gave up trying to {task.instruction}: it was taking too long.{_OWN_FAILURE}"
            )
        case TaskFailed():
            return f"You could not {task.instruction}: {task.status.reason}.{_OWN_FAILURE}"
        case TaskCancelled():
            return f"You stopped before you could {task.instruction}."
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
            return (
                f"You gave up trying to {describe(action)}: it was taking too long.{_OWN_FAILURE}"
            )
        case TaskFailed():
            return f"You could not {describe(action)}: {motion.status.reason}.{_OWN_FAILURE}"
        case TaskCancelled():
            return f"You stopped before you could {describe(action)}."
        case TaskRunning():
            raise AssertionError(f"Motion {motion.id} reported as ended while running")
        case _:
            assert_never(motion.status)


def _speaker(speaker: UtteranceSpeaker | None, permission: ActionPermission) -> Speaker | None:
    match speaker:
        case None | NobodySeenSpeaking():
            return None
        case SeenSpeaking(person=person):
            return Speaker(
                track_id=speaker.track_id,
                person=person,
                operator=person is not None and permission.is_operator(person.user_id),
            )
        case _:
            assert_never(speaker)


def _said_to_robot(speaker: UtteranceSpeaker | None) -> bool:
    """Whether to hear an utterance: not when it was said to someone else, or by nobody in view."""
    match speaker:
        case None:
            return True
        case SeenSpeaking():
            return speaker.to_robot
        case NobodySeenSpeaking():
            return False
        case _:
            assert_never(speaker)


def _name(track_id: int, person: KnownPerson | None) -> str:
    if person is None:
        return f"The unknown person on track {track_id}"
    return f"{person.name} (track {track_id})"


def _describe_person(
    track_id: int, person: KnownPerson | None, permission: ActionPermission
) -> str:
    if person is None:
        return f"An unknown person (track {track_id})"
    # Said up front so the LLM declines a non-operator's request instead of trying it.
    role = "one of" if permission.is_operator(person.user_id) else "not one of"
    return f"{person.name} (track {track_id}, user {person.user_id}, {role} your operators)"


def _normalized(text: str) -> str:
    return " ".join(_PUNCTUATION.sub("", text).lower().split())


def _mentions(text: str, phrases: frozenset[str]) -> bool:
    """Whether any of `phrases` appears in `text` as whole words, ignoring case and punctuation."""
    padded = f" {_normalized(text)} "
    return any(f" {_normalized(phrase)} " in padded for phrase in phrases)


async def _silently(sentences: AsyncIterator[str]) -> Spoken:
    """Consume the LLM's sentences without speaking them."""
    async for _ in sentences:
        pass
    return Spoken(text="", end="finished")


def _always() -> bool:
    return True


async def _single(text: str) -> AsyncIterator[str]:
    yield text


async def _no_events() -> AsyncIterator[Never]:
    return
    yield
