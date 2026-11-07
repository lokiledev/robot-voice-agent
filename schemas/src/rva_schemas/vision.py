from typing import Annotated, Final, Literal
from uuid import UUID

from pydantic import Field

from rva_pydantic import RVABaseModel


class BoundingBox(RVABaseModel):
    """A box in image pixels, from its top-left corner."""

    x: float
    y: float
    width: float
    height: float


class Keypoint(RVABaseModel):
    """A point of a skeleton, in image pixels."""

    x: float
    y: float
    score: float
    """The pose model's confidence."""


class Hand(RVABaseModel):
    keypoints: tuple[Keypoint | None, ...]
    """21 points in COCO-WholeBody order: the wrist, then each finger from the thumb, from its base
    to its tip; `None` where the model is not confident."""


class Body(RVABaseModel):
    """The skeleton of one person."""

    box: BoundingBox
    """Where the person was detected."""
    keypoints: tuple[Keypoint | None, ...]
    """23 points in COCO-WholeBody order: the 17 of COCO (nose, eyes, ears, shoulders, elbows,
    wrists, hips, knees, ankles, each pair left first), then the big toe, small toe and heel of the
    left foot and of the right; `None` where the model is not confident."""
    left_hand: Hand
    """The person's own left."""
    right_hand: Hand


BODY_CONNECTIONS: Final = (
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
)  # fmt: skip
"""Bones between `Body.keypoints`, as index pairs."""
FINGERS: Final = tuple((0, base, base + 1, base + 2, base + 3) for base in (1, 5, 9, 13, 17))
"""`Hand.keypoints` of each finger, from the thumb to the little finger: the wrist, then the finger
from its base to its tip."""


type Expression = Literal["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class Gaze(RVABaseModel):
    """Where the eyes point, in degrees; both are 0 when looking straight at the camera."""

    yaw: float
    """Positive towards the right of the image."""
    pitch: float
    """Positive upwards."""


class FaceFeatures(RVABaseModel):
    """What a face shows in one frame."""

    landmarks: tuple[tuple[float, float], ...]
    """The 478 points of MediaPipe's face mesh, in image pixels; 468 to 477 are the irises."""
    left_eye_open: float
    """From 0, closed, to 1, open; the person's own left."""
    right_eye_open: float
    mouth_open: float
    """From 0, closed, to 1, wide open."""
    gaze: Gaze
    expression: Expression
    expression_score: float
    """Probability of `expression`, from 0 to 1."""


class KnownPerson(RVABaseModel):
    """Someone the robot has been told the name of."""

    user_id: UUID
    name: str
    """Unique among known people."""


class VisionTimings(RVABaseModel):
    """Smoothed per-frame figures of the camera pipeline, in milliseconds but for `fps`."""

    fps: float
    capture_ms: float
    detect_ms: float
    embed_ms: float
    faces_ms: float
    bodies_ms: float
    """Time left waiting for the skeletons once the faces are done: they run alongside."""


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
