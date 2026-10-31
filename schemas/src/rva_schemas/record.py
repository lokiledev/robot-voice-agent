from typing import Annotated, Literal

from pydantic import Field

from rva_body_schemas.motion import Motion
from rva_pydantic import RVABaseModel
from rva_schemas.vision import BoundingBox, FaceEvent, KnownPerson
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


class MotionUpdate(RVABaseModel):
    kind: Literal["motion"] = "motion"
    motion: Motion


class Measurement(RVABaseModel):
    """A sampled value to plot over time, e.g. the voice activity probability."""

    kind: Literal["measurement"] = "measurement"
    name: str
    value: float


class FaceUpdate(RVABaseModel):
    kind: Literal["face"] = "face"
    event: FaceEvent


class FaceBox(RVABaseModel):
    track_id: int
    box: BoundingBox
    person: KnownPerson | None
    confirmed: bool
    """Whether the face was seen long enough to be announced."""
    said: str | None
    """What this person said last, while it is still shown."""


class FaceBoxes(RVABaseModel):
    """Every face tracked in one camera frame."""

    kind: Literal["face_boxes"] = "face_boxes"
    image_width: int
    image_height: int
    """Size of the image the boxes are in: the recorded video's when there is one."""
    boxes: tuple[FaceBox, ...]


class CameraFrame(RVABaseModel):
    kind: Literal["camera_frame"] = "camera_frame"
    jpeg: bytes


type Record = Annotated[
    DialogueLine
    | ToolInvocation
    | TaskUpdate
    | MotionUpdate
    | Measurement
    | FaceUpdate
    | FaceBoxes
    | CameraFrame,
    Field(discriminator="kind"),
]
