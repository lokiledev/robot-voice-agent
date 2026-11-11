from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from rva_pydantic import RVABaseModel
from rva_schemas.vision import KnownPerson


class SpeechSpan(RVABaseModel):
    """When an utterance was spoken, to look up what was seen meanwhile.

    In seconds of the monotonic clock the camera's frames are stamped with.
    """

    start: float
    end: float
    """When the speech ended, before the silence that ended the turn."""


class SeenSpeaking(RVABaseModel):
    """The face in view that said an utterance."""

    kind: Literal["seen"] = "seen"
    track_id: int
    person: KnownPerson | None
    """`None` for someone the robot does not know."""
    to_robot: bool
    """Whether they said it to the robot, rather than to someone else in view."""


class NobodySeenSpeaking(RVABaseModel):
    """No face in view was speaking while the utterance was heard."""

    kind: Literal["unseen"] = "unseen"


type UtteranceSpeaker = Annotated[SeenSpeaking | NobodySeenSpeaking, Field(discriminator="kind")]


class UserUtterance(RVABaseModel):
    """One turn of the user, as recognized text."""

    text: str
    started_at: datetime
    ended_at: datetime
    speech: SpeechSpan | None
    """`None` for a typed line."""
    speaker: UtteranceSpeaker | None
    """Who said it, where the camera and the microphone were put together; `None` where they
    were not, and for a typed line."""


type SpeechEnd = Literal["finished", "talked_over", "cut_short"]
"""How the robot stopped speaking: it said everything, the user talked over it, or it stopped
after a sentence because it was told not to keep going."""


class Spoken(RVABaseModel):
    """What the robot actually said out loud."""

    text: str
    """The text up to where playback stopped."""
    end: SpeechEnd


class ToolCall(RVABaseModel):
    kind: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str
    """JSON-encoded arguments, as produced by the LLM; validated by the tool."""


class ToolSpec(RVABaseModel):
    name: str
    description: str
    parameters: dict[str, JsonValue]
    """JSON schema of the arguments."""


class TextDelta(RVABaseModel):
    kind: Literal["text"] = "text"
    text: str


class TextEnd(RVABaseModel):
    """No more text follows in this turn: only tool calls, still being written."""

    kind: Literal["text_end"] = "text_end"


type LlmOutput = Annotated[TextDelta | TextEnd | ToolCall, Field(discriminator="kind")]


class SystemMessage(RVABaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(RVABaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(RVABaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: tuple[ToolCall, ...]


class ToolResultMessage(RVABaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


class RobotEventMessage(RVABaseModel):
    """Something that happened to the robot outside the conversation, e.g. a task finished.

    Chat templates differ in where they accept system messages, so each LLM backend decides how to
    render it.
    """

    role: Literal["robot_event"] = "robot_event"
    content: str


type Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | RobotEventMessage,
    Field(discriminator="role"),
]
