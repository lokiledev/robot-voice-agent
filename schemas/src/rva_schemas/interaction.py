from typing import Annotated, Literal

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.vision import KnownPerson


class Engaged(RVABaseModel):
    """Someone in view started paying attention to the robot: close, staying, looking at it."""

    kind: Literal["engaged"] = "engaged"
    track_id: int
    person: KnownPerson | None


class Disengaged(RVABaseModel):
    """Someone engaged no longer pays attention to the robot."""

    kind: Literal["disengaged"] = "disengaged"
    track_id: int
    person: KnownPerson | None
    left: bool
    """Whether they went out of view, rather than turned or walked away in view."""


type EngagementEvent = Annotated[Engaged | Disengaged, Field(discriminator="kind")]


class PersonEngagement(RVABaseModel):
    """How much one person in view pays attention to the robot, in one frame."""

    track_id: int
    closeness: float
    """1 within reach of a conversation, 0 from the configured distance on."""
    staying: float
    """1 standing still, 0 from a walking pace on."""
    attending: bool
    """Whether they look at the robot or, once engaged, speak."""
    score: float
    """From 0 to 1: `closeness * staying` while `attending`, 0 otherwise, smoothed over time."""
    engaged: bool


class EngagementState(RVABaseModel):
    """Who pays attention to the robot in one frame, and who started or stopped to."""

    kind: Literal["engagement"] = "engagement"
    time: float
    """The observation's time."""
    people: tuple[PersonEngagement, ...]
    """Every tracked face."""
    events: tuple[EngagementEvent, ...]
