import asyncio
import json
import re
import time
import unicodedata
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from datetime import UTC, datetime
from typing import Final, Never, assert_never
from uuid import UUID, uuid4

from loguru import logger
from opentelemetry import trace

from rva_agent.body import BodySupervisor, BodyTools
from rva_agent.config import AgentConfig
from rva_agent.history import History
from rva_agent.people import PeopleTools
from rva_agent.permission import ActionPermission
from rva_agent.prompt import render_system_prompt
from rva_agent.sentences import StageDirections, split_sentences
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import TaskTools, ToolResult
from rva_body_schemas.abc.client import BodyClient
from rva_body_schemas.motion import GoToHuman, GoToLocation, Motion, PlayAnimation, describe
from rva_pydantic import RVABaseModel
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
    TextEnd,
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
    RobotReplyEnded,
    Speaker,
    ToolInvocation,
    UserLine,
)
from rva_schemas.vision import FaceAppeared, FaceEvent, FaceLeft, KnownPerson
from rva_vla_schemas.abc.client import VlaClient
from rva_vla_schemas.task import Task, TaskCancelled, TaskFailed, TaskRunning, TaskSucceeded

_tracer: Final = trace.get_tracer("rva_agent")
_PUNCTUATION: Final = re.compile(r"[^\w\s]")
_WORD: Final = re.compile(r"\w+")
# Told with each failure: deep in a long conversation, the LLM otherwise reports some of them from
# the outside ("the task to gather the cables failed") despite its instructions.
_OWN_FAILURE: Final = ' Say it as your own failure, like "I couldn\'t ..."; never call it a task.'
# Asked, unheard, after a turn of someone unknown in which nothing was remembered: Qwen skips the
# call about once in ten, and a reply talked over never gets to it, as when a question comes in
# a breath with the name ("I'm Ben, can you tell us a story?").
_NAME_CHECK: Final = (
    "Did the unknown person on track {track_id} just say what their own name is? Answer with"
    " their name as they said it, or null if they did not say their name. A place, a feeling or"
    " someone else's name is not their name. Nobody hears this."
)
# Before a line said while the robot's reply to something else was on its way: such as "Someone's
# coming" as it greets the newcomer, which Qwen answered with the same greeting 8 times in 10, and
# still 5 in 10 when only told the line came before its reply.
_SAID_BEFORE_REPLY: Final = (
    "(Said before you finished your last reply: that reply already answered it, so do not say it"
    " again. Answer only something new, or say nothing.)"
)
_REPEATED: Final = ToolResult(
    content='{"error": "You just called this, and it was not run again. Do not call it again now."}',
    needs_reply=False,
)


class _NameTold(RVABaseModel):
    name: str | None
    """Their own name, as they said it; `None` when they did not say it."""


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
        capabilities = profile.capabilities
        self._body_tools: Final = BodyTools(
            self._body,
            self._supervisor,
            self._permission,
            vision,
            tuple(place.name for place in profile.environment.places),
            capabilities,
        )
        self._people: Final = None if vision is None else PeopleTools(vision, self._permission)
        self._muted = False
        self._tool_specs: Final = (
            # A robot without a VLA has no tasks to start.
            (TaskTools.specs if capabilities.skills else ())
            + self._body_tools.specs
            + (() if self._people is None else PeopleTools.specs)
        )
        self._tool_names: Final = frozenset(spec.name for spec in self._tool_specs)
        self._greeting_animation: Final = capabilities.greeting_animation
        self._history: Final = History(
            render_system_prompt(profile, vision=vision is not None), config.max_history_messages
        )
        # When each known person last disengaged, by user id, on the monotonic clock.
        self._disengaged_at: Final[dict[UUID, float]] = {}
        # The tracks already greeted or heard from: a face is greeted once.
        self._greeted: Final[set[int]] = set()
        # The track of whoever stopped listening in the middle of a reply, until they are back.
        self._cut_short_for: int | None = None
        # Who they are, if known: out of view and back, they are on a new track.
        self._cut_short_person: UUID | None = None
        # Whether the robot said something that no `RobotReplyEnded` followed yet.
        self._replying = False
        # When the robot last stopped saying words out loud, on the utterances' clock.
        self._said_until: datetime | None = None
        # What was said to the robot lately, as heard: a name it remembers is one of these words.
        self._heard: Final = deque[str](maxlen=config.max_history_messages)

    async def warm_up(self) -> None:
        """Have the LLM read the system prompt and the tools ahead of the first turn.

        A server that reads them for the first time takes seconds to start its reply: the greeting
        of whoever comes first.
        """
        # Chat templates such as Qwen's want a user message; what is read ahead is all before it.
        messages = (*self._history.messages, UserMessage(content="Hello."))
        # The prompt is read by the first output; the rest of the reply is not needed.
        async for _ in self._llm.stream(messages, self._tool_specs):
            break

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
                # Every round of the reply is said by now; with nothing else waiting for its
                # turn, the robot has no more to say until something new happens.
                if self._replying and self._inbox.empty():
                    self._replying = False
                    self._recorder.record(RobotReplyEnded())
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
        self._heard.append(utterance.text)
        self._recorder.record(
            UserLine(text=utterance.text, speaker=_speaker(speaker, self._permission))
        )
        self._history.append(UserMessage(content=self._attributed(utterance, speaker)))
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
        called = await self._respond(listener)
        if (
            isinstance(speaker, SeenSpeaking)
            and self._unknown_in_view(speaker.track_id)
            and not any(name == "remember_person" for name, _ in called)
        ):
            await self._remember_if_named(speaker.track_id, utterance.text, called)

    async def _on_task_end(self, task: Task) -> None:
        self._history.append(RobotEventMessage(content=_describe_end(task)))
        match task.status:
            case TaskCancelled():
                # Someone asked for it, the user already knows.
                pass
            case TaskSucceeded() | TaskFailed():
                # Only an operator could have it started: so may its next step be.
                with self._permission.going_on():
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
        if self._was_cut_short_for(event):
            self._cut_short_for = self._cut_short_person = None
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
        self._history.append(
            RobotEventMessage(
                content=f"{who} came to you and is looking at you. {self._greeting()}"
            )
        )
        await self._respond(event.track_id)

    def _greeting(self) -> str:
        """What to do when someone comes, with what this body has to do it.

        Said in the event itself, which the model follows far more than a rule in its prompt. Told
        to wave without an animation for it, it writes the wave into what it says.
        """
        if self._greeting_animation is None:
            return "Greet them."
        return f"Greet them: play your {self._greeting_animation} animation."

    def _was_cut_short_for(self, event: Engaged) -> bool:
        """Whether the reply that stopped in the middle was to whoever engages now."""
        if event.track_id == self._cut_short_for:
            return True
        return event.person is not None and event.person.user_id == self._cut_short_person

    def _on_disengaged(self, event: Disengaged) -> None:
        if event.person is not None:
            self._disengaged_at[event.person.user_id] = time.monotonic()
            if event.track_id == self._cut_short_for:
                self._cut_short_person = event.person.user_id
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
            case _, PlayAnimation():
                # Anyone may ask for a gesture: its failure opens nothing more to them.
                await self._respond(None)
            case _, GoToLocation() | GoToHuman():
                # Only an operator could have it started: so may its next step be.
                with self._permission.going_on():
                    await self._respond(None)
            case _:
                assert_never(motion.action)

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

    def _attributed(self, utterance: UserUtterance, speaker: UtteranceSpeaker | None) -> str:
        """The text of `utterance`, prefixed with what the LLM needs to know about it.

        Who said it, what the body is busy with, if anything, and whether it was said before the
        robot's last reply was over. Without these, the LLM keeps answering whoever spoke last
        after someone else speaks, promises an action before finding out the body is busy, and
        answers again what its last reply answered: a line said while that reply was on its way
        comes after it in the history.
        """
        text = utterance.text
        task = self._supervisor.active
        busy = task.instruction if task is not None else self._moving()
        if busy is not None:
            text = f"(Your body is busy: {busy}.) {text}"
        if self._said_until is not None and utterance.ended_at < self._said_until:
            text = f"{_SAID_BEFORE_REPLY} {text}"
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

    async def _respond(self, listener: int | None) -> set[tuple[str, str]]:
        """Let the LLM speak and call tools until it has nothing left to do.

        `listener` is the track of the person answered, if it is one person: when they are engaged
        as the reply starts and stop being so, the reply stops after its sentence.

        Returns:
            The calls run, by name and arguments.
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
                        _silently(sentences) if muted else self._say(sentences, keep_going)
                    )
            if muted:
                # Nothing was heard, so nothing was said.
                text.clear()
            match spoken.end:
                case "talked_over":
                    # Keep what the user heard, drop the rest and any action: the reply stops
                    # being written there. A name it would have remembered is checked for after.
                    self._add_robot_message(
                        AssistantMessage(content=spoken.text, tool_calls=()), interrupted=True
                    )
                    return called
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
                    return called
                case "finished":
                    pass
                case _:
                    assert_never(spoken.end)
            self._add_robot_message(
                AssistantMessage(content="".join(text), tool_calls=tuple(tool_calls)),
                interrupted=False,
            )
            if not tool_calls:
                return called
            needs_reply = False
            for call in tool_calls:
                result = await self._run_tool(call, called)
                needs_reply = needs_reply or result.needs_reply
            # Asking the LLM again here would only make it repeat what it just announced, or, with
            # nothing left to say, make something up: an event it expects, said aloud.
            if not needs_reply:
                return called
        logger.warning(
            "The LLM was still calling tools after {} rounds", self._config.max_tool_rounds
        )
        return called

    def _unknown_in_view(self, track_id: int) -> bool:
        """Whether the face on `track_id` is in view and nobody known, now.

        Not as the utterance was attributed: a name remembered meanwhile, while the next line was
        heard, counts.
        """
        if self._people is None or self._vision is None:
            return False
        face = next((face for face in self._vision.faces() if face.track_id == track_id), None)
        return face is not None and face.person is None

    async def _remember_if_named(
        self, track_id: int, said: str, called: set[tuple[str, str]]
    ) -> None:
        """Ask the LLM, unheard, whether the unknown person on `track_id` told their name.

        It can only answer with a name or none, and only a name found in `said` is remembered:
        asked, the LLM sometimes makes a name up, such as its prompt's example. The question and
        the answer join the history only when a name was remembered, as a call to
        remember_person.

        Args:
            track_id: Whoever just spoke, unknown.
            said: What they said.
            called: The calls of this turn so far; the one made here joins them.
        """
        check = RobotEventMessage(content=_NAME_CHECK.format(track_id=track_id))
        with _tracer.start_as_current_span("llm:name_check"):
            told = await self._llm.answer(
                (*self._history.messages, check), self._tool_specs, _NameTold
            )
        name = told.name
        if name is not None and not _names(said, name):
            logger.warning("The name check made up a name not in {!r}: {}", said, name)
            name = None
        logger.info("Checked whether track {} told their name: {}", track_id, name or "no")
        if name is None:
            return
        call = ToolCall(
            id=f"name_check_{uuid4().hex[:8]}",
            name="remember_person",
            arguments=json.dumps({"track_id": track_id, "name": name}),
        )
        self._history.append(check)
        self._history.append(AssistantMessage(content="", tool_calls=(call,)))
        await self._run_tool(call, called)

    async def _run_tool(self, call: ToolCall, called: set[tuple[str, str]]) -> ToolResult:
        """Run a call of this turn, recorded with its result, which goes into the history.

        Args:
            call: As the LLM wrote it.
            called: The calls of this turn so far, by name and arguments; `call` joins them.
        """
        if (call.name, call.arguments) in called:
            # The LLM retries what just failed, round after round, saying each time that it does
            # it now: the turn ends here instead.
            result = _REPEATED
        else:
            called.add((call.name, call.arguments))
            with _tracer.start_as_current_span(f"tool:{call.name}"):
                result = await self._call_tool(call)
        self._recorder.record(
            ToolInvocation(name=call.name, arguments=call.arguments, result=result.content)
        )
        self._history.append(ToolResultMessage(tool_call_id=call.id, content=result.content))
        return result

    def _while_listening(self, listener: int | None) -> Callable[[], bool]:
        """Whether to keep talking: until a `listener` engaged now no longer is."""
        engagement = self._engagement
        if engagement is None or listener is None or listener not in engagement.engaged():
            return _always
        return lambda: listener in engagement.engaged()

    async def _call_tool(self, call: ToolCall) -> ToolResult:
        if call.name not in self._tool_names:
            # One of the tools this robot's body does not have, or none at all.
            return ToolResult(
                content=json.dumps({"error": f"There is no tool named {call.name}."}),
                needs_reply=True,
            )
        if (
            call.name == "remember_person"
            and (name := _name_given(call)) is not None
            and not any(_names(heard, name) for heard in self._heard)
        ):
            # The LLM now and then remembers someone under a name of its own: its prompt's
            # example, or "unknown person". The first person a fresh robot knows is an operator.
            logger.warning("Not remembered, a name nobody said: {}", name)
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"Nobody told you the name {name}. Never guess a name: ask"
                        " them theirs, if you have not yet."
                    }
                ),
                needs_reply=True,
            )
        if self._people is not None and call.name in PeopleTools.names:
            return await self._people.call(call)
        if call.name in BodyTools.names:
            return await self._body_tools.call(call)
        return await self._tools.call(call)

    async def _stream_text(
        self, text: list[str], tool_calls: list[ToolCall]
    ) -> AsyncGenerator[str]:
        span = trace.get_current_span()
        # Out of the history too: the LLM would write one in every reply after.
        directions = StageDirections()
        try:
            async for output in self._llm.stream(self._history.messages, self._tool_specs):
                match output:
                    case TextDelta():
                        if not text:
                            span.add_event("first_text")
                        said = directions.strip(output.text)
                        text.append(said)
                        yield said
                    case TextEnd():
                        # Ends the last sentence, which is said while the tool calls are written.
                        yield "\n"
                    case ToolCall():
                        tool_calls.append(output)
                    case _:
                        assert_never(output)
        finally:
            if directions.removed:
                logger.info("Not said, from the reply: stage directions {}", directions.removed)

    async def _say_fixed(self, text: str) -> None:
        if self._muted:
            return
        spoken = await self._say(_single(text), _always)
        self._add_robot_message(
            AssistantMessage(content=spoken.text, tool_calls=()),
            interrupted=spoken.end != "finished",
        )

    async def _say(self, sentences: AsyncIterator[str], keep_going: Callable[[], bool]) -> Spoken:
        spoken = await self._speech.say(sentences, keep_going)
        if spoken.text.strip():
            self._said_until = datetime.now(UTC)
        return spoken

    def _add_robot_message(self, message: AssistantMessage, *, interrupted: bool) -> None:
        self._history.append(message)
        # Interrupted in its first sentence, a reply has no words heard, but one was started.
        if message.content.strip() or interrupted:
            self._recorder.record(RobotLine(text=message.content, interrupted=interrupted))
            self._replying = True

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


def _name_given(call: ToolCall) -> str | None:
    """The name a remember_person call gives; `None` when its arguments have none."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    name = arguments.get("name") if isinstance(arguments, dict) else None
    return name if isinstance(name, str) else None


def _names(text: str, name: str) -> bool:
    """Whether `name` is among the words of `text`, however each is spaced, hyphenated or accented.

    "Jean-Luc" is in "my name is jean luc", and "Chloe" in "I'm Chloé"; "Ben" is not in "bench".
    """
    wanted = "".join(_words(name))
    words = _words(text)
    return bool(wanted) and any(
        "".join(words[first:last]) == wanted
        for first in range(len(words))
        for last in range(first + 1, len(words) + 1)
    )


def _words(text: str) -> list[str]:
    """The words of `text`, without case, accents or punctuation."""
    plain = "".join(
        letter
        for letter in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(letter)
    )
    return [match.group() for match in _WORD.finditer(plain)]


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
