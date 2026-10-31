from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from rva_pydantic import RVABaseModel


class UserUtterance(RVABaseModel):
    """One turn of the user, as recognized text."""

    text: str
    started_at: datetime
    ended_at: datetime


class Spoken(RVABaseModel):
    """What the robot actually said out loud."""

    text: str
    """The text up to where playback stopped."""
    interrupted: bool
    """Whether the user talked over the robot and cut it off."""


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


type LlmOutput = Annotated[TextDelta | ToolCall, Field(discriminator="kind")]


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
