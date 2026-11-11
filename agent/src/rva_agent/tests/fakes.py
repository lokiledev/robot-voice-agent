import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final, assert_never
from uuid import NAMESPACE_URL, UUID, uuid5

from rva_schemas.dialogue import (
    LlmOutput,
    Message,
    SpeechEnd,
    Spoken,
    ToolSpec,
    UserUtterance,
    UtteranceSpeaker,
)
from rva_schemas.interaction import Disengaged, Engaged, EngagementEvent
from rva_schemas.profile import Admins, Capabilities, Character, Environment, Profile, Skill
from rva_schemas.record import Record
from rva_schemas.vision import (
    FaceEvent,
    KnownPerson,
    NameTakenError,
    NotRecognizedError,
    Remembered,
    TrackedFace,
    UnknownPersonError,
    UnknownTrackError,
)
from rva_vla_mock.body import BodyMock
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner


class ScriptedLanguageModel:
    """Answers each call with the next scripted turn and keeps the messages it was given."""

    def __init__(self, turns: Sequence[Sequence[LlmOutput]]) -> None:
        self._turns: Final = list(turns)
        self.requests: Final[list[tuple[Message, ...]]] = []

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncGenerator[LlmOutput]:
        self.requests.append(tuple(messages))
        if not self._turns:
            raise AssertionError(f"Unexpected LLM call after: {messages[-1]!r}")
        for output in self._turns.pop(0):
            yield output


class QueueUserInput:
    def __init__(self) -> None:
        self._lines: Final = asyncio.Queue[tuple[str, UtteranceSpeaker | None] | None]()

    def say(self, text: str) -> None:
        self._lines.put_nowait((text, None))

    def say_as(self, text: str, speaker: UtteranceSpeaker) -> None:
        """Say `text` as an utterance the camera and the microphone attributed to `speaker`."""
        self._lines.put_nowait((text, speaker))

    def leave(self) -> None:
        self._lines.put_nowait(None)

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        while (line := await self._lines.get()) is not None:
            text, speaker = line
            now = datetime.now(UTC)
            yield UserUtterance(
                text=text, started_at=now, ended_at=now, speech=None, speaker=speaker
            )


class RecordingSpeech:
    """Collects what the robot says; interrupts it after a number of sentences if asked to."""

    def __init__(self, interrupt_after: int | None) -> None:
        self._interrupt_after: Final = interrupt_after
        self.said: Final[list[str]] = []
        self.interruptions = 0
        self.after_sentence: Callable[[str], None] | None = None
        """Called once each sentence is said, e.g. for the listener to turn away then."""
        self._changed: Final = asyncio.Condition()

    async def say(self, sentences: AsyncIterator[str], keep_going: Callable[[], bool]) -> Spoken:
        heard: list[str] = []
        end: SpeechEnd = "finished"
        async for sentence in sentences:
            if not keep_going():
                end = "cut_short"
                break
            heard.append(sentence)
            if self.after_sentence is not None:
                self.after_sentence(sentence)
            if len(heard) == self._interrupt_after:
                end = "talked_over"
                break
        async with self._changed:
            self.said.extend(heard)
            self._changed.notify_all()
        return Spoken(text=" ".join(heard), end=end)

    def interrupt(self) -> None:
        self.interruptions += 1

    async def wait_for(self, sentence: str) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: sentence in self.said)


class ListRecorder:
    def __init__(self) -> None:
        self.entries: Final[list[Record]] = []
        self._changed: Final = asyncio.Condition()

    def record(self, entry: Record) -> None:
        self.entries.append(entry)
        asyncio.get_running_loop().create_task(self._notify())

    async def wait_until(self, predicate: Callable[[list[Record]], bool]) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: predicate(self.entries))

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()


class FakeEngagement:
    """Engages and disengages the people a test says."""

    def __init__(self, engaged: Sequence[int]) -> None:
        """Start with the people on the `engaged` tracks already engaged."""
        self._events: Final = asyncio.Queue[EngagementEvent]()
        self._engaged: Final = set(engaged)

    def show(self, event: EngagementEvent) -> None:
        match event:
            case Engaged():
                self._engaged.add(event.track_id)
            case Disengaged():
                self._engaged.discard(event.track_id)
            case _:
                assert_never(event)
        self._events.put_nowait(event)

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[EngagementEvent]]:
        yield self._drain()

    def engaged(self) -> frozenset[int]:
        return frozenset(self._engaged)

    async def _drain(self) -> AsyncIterator[EngagementEvent]:
        while True:
            yield await self._events.get()


def person(name: str) -> KnownPerson:
    return KnownPerson(user_id=uuid5(NAMESPACE_URL, name), name=name)


class FakeVision:
    """Shows the faces and events a test sets, and remembers names in memory."""

    def __init__(self, faces: Sequence[TrackedFace]) -> None:
        self._faces: Final = list(faces)
        self._events: Final = asyncio.Queue[FaceEvent]()
        self.known: Final[dict[str, KnownPerson]] = {}
        self.granted: Final[dict[UUID, UUID]] = {}
        """Granted operator to the operator who granted them."""

    def show(self, event: FaceEvent) -> None:
        self._events.put_nowait(event)

    def set_faces(self, faces: Sequence[TrackedFace]) -> None:
        """Change who is in view, e.g. someone else stepping in front."""
        self._faces[:] = faces

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[FaceEvent]]:
        yield self._drain()

    def faces(self) -> tuple[TrackedFace, ...]:
        return tuple(self._faces)

    async def remember(self, track_id: int, name: str) -> Remembered:
        index = next((i for i, face in enumerate(self._faces) if face.track_id == track_id), None)
        if index is None:
            raise UnknownTrackError(track_id)
        known = self.known.setdefault(name, person(name))
        previous = self._faces[index].person
        self._faces[index] = self._faces[index].model_copy(update={"person": known})
        return Remembered(person=known, previous=None if previous == known else previous)

    async def forget(self, name: str) -> KnownPerson:
        if name not in self.known:
            raise UnknownPersonError(name)
        return self.known.pop(name)

    async def rename(self, track_id: int, name: str) -> KnownPerson:
        index = next((i for i, face in enumerate(self._faces) if face.track_id == track_id), None)
        if index is None:
            raise UnknownTrackError(track_id)
        current = self._faces[index].person
        if current is None:
            raise NotRecognizedError(track_id)
        if name in self.known and self.known[name].user_id != current.user_id:
            raise NameTakenError(name)
        renamed = KnownPerson(user_id=current.user_id, name=name)
        self.known.pop(current.name, None)
        self.known[name] = renamed
        self._faces[index] = self._faces[index].model_copy(update={"person": renamed})
        return renamed

    async def people(self) -> tuple[KnownPerson, ...]:
        in_view = [face.person for face in self._faces if face.person is not None]
        return tuple({p.user_id: p for p in [*in_view, *self.known.values()]}.values())

    def granted_operators(self) -> frozenset[UUID]:
        return frozenset(self.granted)

    async def grant_operator(self, user_id: UUID, granted_by: UUID) -> None:
        self.granted[user_id] = granted_by

    async def revoke_operator(self, user_id: UUID) -> bool:
        return self.granted.pop(user_id, None) is not None

    async def _drain(self) -> AsyncIterator[FaceEvent]:
        while True:
            yield await self._events.get()


PROFILE = Profile(
    character=Character(
        name="Lune", personality="Calm.", speaking_style="Short.", verbosity="terse"
    ),
    environment=Environment(location="Lab", description="A lab.", places=(), objects=(), people=()),
    capabilities=Capabilities(
        skills=(
            Skill(name="pick", description="Pick up an object.", examples=("pick up the red cup",)),
        ),
        constraints=(),
        animations=("wave_right_hand", "dance"),
    ),
)
ADMINS = Admins(user_ids=frozenset({person("Alice").user_id}))


def idle_body(vla: TaskRunner) -> BodyMock:
    """A body whose motions only end when a test types a command, sharing `vla`'s busy state."""
    idle = TerminalPolicy(asyncio.Queue(), fallback=None)
    return BodyMock(vla, idle, idle)
