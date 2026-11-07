from typing import Annotated, Literal

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.vision import KnownPerson, Point3D


class FaceEntity(RVABaseModel):
    """A face, at the point between its irises."""

    kind: Literal["face"] = "face"
    track_id: int
    person: KnownPerson | None


class HandEntity(RVABaseModel):
    """A hand, at the centre of its palm."""

    kind: Literal["hand"] = "hand"
    track_id: int
    """The face track of its owner."""
    side: Literal["left", "right"]
    """The owner's own left or right."""


type AttentionEntity = Annotated[FaceEntity | HandEntity, Field(discriminator="kind")]


class AttentionFeatures(RVABaseModel):
    """What draws attention to an entity, each from 0 to 1. A hand has only `moving`."""

    closeness: float
    """1 at the camera, 0 from the configured distance on."""
    centrality: float
    """1 on the camera's axis, 0 from the configured angle off it on."""
    speaking: float
    """1 while the active speaker detector hears them speak."""
    eye_contact: float
    """1 when the gaze is on the camera, 0 from the configured angle off it on."""
    moving: float
    """The speed over the last window, 1 from the configured speed on."""
    known: float
    """1 for someone the robot knows by name."""


class AttentionCandidate(RVABaseModel):
    """An entity in view that could be attended, and why."""

    entity: AttentionEntity
    point: Point3D
    features: AttentionFeatures
    novelty: float
    """The bonus for having just appeared, decaying since."""
    habituation: float
    """From 1, fresh, down to the configured floor as it is attended; it recovers while not."""
    score: float
    """The weighted features times the habituation, plus the novelty."""


class AttentionTarget(RVABaseModel):
    entity: AttentionEntity
    point: Point3D
    """In the camera's frame, raw: filtered downstream, where the gaze or the body wants it."""


class AttentionState(RVABaseModel):
    """What the robot attends to in one frame, and everything it could have."""

    kind: Literal["attention"] = "attention"
    time: float
    """The observation's time."""
    target: AttentionTarget | None
    """`None` when nothing in view is worth attending, or nothing is placed in 3D."""
    candidates: tuple[AttentionCandidate, ...]
    """Every entity in view and in 3D, best first."""
