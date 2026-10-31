from typing import Annotated, Literal

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.vision import FaceEvent
from rva_vla_schemas.task import Task


class DialogueLine(RVABaseModel):
    kind: Literal["dialogue"] = "dialogue"
    speaker: Literal["user", "robot"]
    text: str


class ToolInvocation(RVABaseModel):
    kind: Literal["tool"] = "tool"
    name: str
    arguments: str
    result: str


class TaskUpdate(RVABaseModel):
    kind: Literal["task"] = "task"
    task: Task


class Measurement(RVABaseModel):
    """A sampled value to plot over time, e.g. the voice activity probability."""

    kind: Literal["measurement"] = "measurement"
    name: str
    value: float


class FaceUpdate(RVABaseModel):
    kind: Literal["face"] = "face"
    event: FaceEvent


type Record = Annotated[
    DialogueLine | ToolInvocation | TaskUpdate | Measurement | FaceUpdate,
    Field(discriminator="kind"),
]
