from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from rva_pydantic import RVABaseModel


class BoundingBox(RVABaseModel):
    """A box in image pixels, from its top-left corner."""

    x: float
    y: float
    width: float
    height: float


class KnownPerson(RVABaseModel):
    """Someone the robot has been told the name of."""

    user_id: UUID
    name: str
    """Unique among known people."""


class TrackedFace(RVABaseModel):
    """A face the robot has seen for long enough to be sure it is there."""

    track_id: int
    """Stable while the face stays in view, including short occlusions."""
    box: BoundingBox
    person: KnownPerson | None
    """`None` while the face matches nobody known."""
    nearest: bool
    """Whether it is the largest face in view, taken as the person in front of the robot."""


class FaceAppeared(RVABaseModel):
    kind: Literal["appeared"] = "appeared"
    face: TrackedFace


class FaceLeft(RVABaseModel):
    kind: Literal["left"] = "left"
    track_id: int
    person: KnownPerson | None


type FaceEvent = Annotated[FaceAppeared | FaceLeft, Field(discriminator="kind")]


class Remembered(RVABaseModel):
    person: KnownPerson
    previous: KnownPerson | None
    """Who the face was recognized as before, if it was someone else."""


class UnknownTrackError(Exception):
    def __init__(self, track_id: int) -> None:
        super().__init__(f"No face with track {track_id} is in view")
        self.track_id = track_id


class UnknownPersonError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Nobody named {name!r} is known")
        self.name = name


class NameTakenError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Someone else is already called {name!r}")
        self.name = name


class NotRecognizedError(Exception):
    """The face on a track matches nobody known yet."""

    def __init__(self, track_id: int) -> None:
        super().__init__(f"The face on track {track_id} is not recognized")
        self.track_id = track_id
