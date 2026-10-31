from datetime import datetime
from typing import Annotated, Literal, assert_never
from uuid import UUID

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_vla_schemas.task import TaskStatus


class GoToLocation(RVABaseModel):
    kind: Literal["go_to_location"] = "go_to_location"
    location: str
    """A named place of the environment, e.g. "kitchen counter"."""


class GoToHuman(RVABaseModel):
    kind: Literal["go_to_human"] = "go_to_human"
    user_id: UUID


class PlayAnimation(RVABaseModel):
    kind: Literal["animation"] = "animation"
    animation: str
    """e.g. "wave_right_hand"."""


type MotionAction = Annotated[GoToLocation | GoToHuman | PlayAnimation, Field(discriminator="kind")]


class MotionRequest(RVABaseModel):
    action: MotionAction


class Motion(RVABaseModel):
    """A movement of the whole body: walking somewhere or playing an animation."""

    id: UUID
    action: MotionAction
    status: TaskStatus


class MotionEvent(RVABaseModel):
    """A status change of a motion, pushed by the body."""

    motion_id: UUID
    status: TaskStatus
    timestamp: datetime


class GazeRequest(RVABaseModel):
    user_id: UUID
    """The person to keep looking at, turning the head and body without walking."""


class BodyState(RVABaseModel):
    motion: Motion | None
    """The motion running now, if any."""
    looking_at: UUID | None


def describe(action: GoToLocation | GoToHuman | PlayAnimation) -> str:
    match action:
        case GoToLocation():
            return f"go to the {action.location}"
        case GoToHuman():
            return f"go to user {action.user_id}"
        case PlayAnimation():
            return f"play the {action.animation} animation"
        case _:
            assert_never(action)
