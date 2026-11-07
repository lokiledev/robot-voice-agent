from typing import Annotated, Literal

from pydantic import Field

from rva_body_schemas.motion import Motion
from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import ToolSpec
from rva_schemas.vision import Body, BoundingBox, FaceEvent, FaceFeatures, KnownPerson
from rva_vla_schemas.task import Task


class AgentStarted(RVABaseModel):
    kind: Literal["agent_started"] = "agent_started"
    name: str
    """The robot's name."""
    tools: tuple[ToolSpec, ...]
    """The tools offered to the LLM."""


class Speaker(RVABaseModel):
    """The face in front of the robot when a user line was said, taken as who said it."""

    track_id: int
    person: KnownPerson | None
    """`None` for someone the robot does not know."""
    operator: bool
    """Whether they may make the robot act."""


class UserLine(RVABaseModel):
    kind: Literal["user_line"] = "user_line"
    text: str
    speaker: Speaker | None
    """`None` when nobody was in view, or without a camera."""


class RobotSentence(RVABaseModel):
    """A sentence the robot starts saying; its whole reply follows as a `RobotLine`."""

    kind: Literal["robot_sentence"] = "robot_sentence"
    text: str


class RobotLine(RVABaseModel):
    kind: Literal["robot_line"] = "robot_line"
    text: str
    """What the user heard of the reply."""
    interrupted: bool
    """Whether the user talked over the robot, which stopped there."""


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


class GazeUpdate(RVABaseModel):
    kind: Literal["gaze"] = "gaze"
    looking_at: KnownPerson | None
    """The person the robot keeps looking at, or `None` once it stops."""


class MuteUpdate(RVABaseModel):
    kind: Literal["mute"] = "mute"
    muted: bool


class Measurement(RVABaseModel):
    """A sampled value to plot over time, e.g. the voice activity probability."""

    kind: Literal["measurement"] = "measurement"
    name: str
    value: float


class FaceUpdate(RVABaseModel):
    kind: Literal["face"] = "face"
    event: FaceEvent


class TrackedPerson(RVABaseModel):
    """A face track, with what was seen of that person in one frame."""

    track_id: int
    box: BoundingBox
    """The face's last box."""
    person: KnownPerson | None
    confirmed: bool
    """Whether the face was seen long enough to be announced."""
    said: str | None
    """What this person said last, while it is still shown."""
    face: FaceFeatures | None
    """`None` when the face is out of sight in this frame, or face features are turned off."""
    body: Body | None
    """`None` when no skeleton was found under this face, or poses are turned off."""


class PeopleInView(RVABaseModel):
    """Every person tracked or seen in one camera frame."""

    kind: Literal["people"] = "people"
    image_width: int
    image_height: int
    """Size of the image the coordinates are in: the recorded video's when there is one."""
    people: tuple[TrackedPerson, ...]
    unmatched_bodies: tuple[Body, ...]
    """Skeletons with no tracked face: turned away, or too far away to be tracked."""


class CameraFrame(RVABaseModel):
    kind: Literal["camera_frame"] = "camera_frame"
    jpeg: bytes


type Record = Annotated[
    AgentStarted
    | UserLine
    | RobotSentence
    | RobotLine
    | ToolInvocation
    | TaskUpdate
    | MotionUpdate
    | GazeUpdate
    | MuteUpdate
    | Measurement
    | FaceUpdate
    | PeopleInView
    | CameraFrame,
    Field(discriminator="kind"),
]
