from typing import Annotated, Final, Literal
from uuid import UUID

import numpy as np
import numpy.typing as npt
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
    """The 13 upper-body points of COCO: nose, eyes, ears, shoulders, elbows, wrists and hips, each
    pair left first; `None` where the model is not confident. Legs and feet are left out."""
    left_hand: Hand | None
    """The person's own left; `None` when not found, or when it is the hand the right one is on."""
    right_hand: Hand | None
    """`None` when not found, or when it is the hand the left one is on."""


BODY_CONNECTIONS: Final = (
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
)  # fmt: skip
"""Bones between `Body.keypoints`, as index pairs."""
FINGERS: Final = tuple((0, base, base + 1, base + 2, base + 3) for base in (1, 5, 9, 13, 17))
"""`Hand.keypoints` of each finger, from the thumb to the little finger: the wrist, then the finger
from its base to its tip."""


type Expression = Literal["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class Gaze(RVABaseModel):
    """Where the eyes point, in degrees; both are 0 when looking straight at the camera.

    Wherever the face is in the image: the angles are taken from the line between the face and the
    camera, not from the camera's axis: in a recording with the face from 28 degrees left of the
    axis to 32 right, the yaw changed by 0.06 degrees per degree the face moved, where angles from
    the axis would change by 1.
    """

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


class MainUser(RVABaseModel):
    """The person the robot attends to: the closest, most central face in view."""

    track_id: int
    gaze_off_camera: float
    """Angle between the gaze and the camera, in degrees, as a median over the last few frames."""
    eye_contact: bool
    """Whether they look at the robot's camera, held through blinks."""


class KnownPerson(RVABaseModel):
    """Someone the robot has been told the name of."""

    user_id: UUID
    name: str
    """Unique among known people."""


class DepthMap(RVABaseModel, arbitrary_types_allowed=True):
    """Depth of a camera frame, at a resolution of its own with the frame's aspect ratio."""

    metres: npt.NDArray[np.float32]
    """Distance along the camera's axis, height x width."""
    calibrated: bool
    """Whether the camera's measured focal length scaled it. Otherwise an assumed one did, the
    datasheet's or the model's own, so distances may be off by tens of percent."""


class VisionTimings(RVABaseModel):
    """Smoothed per-frame figures of the camera pipeline, in milliseconds but for `fps`."""

    fps: float
    capture_ms: float
    detect_ms: float
    embed_ms: float
    faces_ms: float
    bodies_ms: float
    """Time left waiting for the skeletons once the faces are done: they run alongside."""
    depth_ms: float
    """Time left waiting for the depth once the skeletons are done: it runs alongside too."""


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
