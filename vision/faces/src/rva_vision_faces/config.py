from pathlib import Path
from typing import Literal

from pydantic import Field

from rva_pydantic import RVABaseModel


class ModelFile(RVABaseModel):
    """An ONNX file on the Hugging Face Hub, downloaded on first use."""

    repo: str
    file: str


class CameraConfig(RVABaseModel):
    device: Path
    """V4L2 device, e.g. `/dev/video0`."""
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)


class DetectorConfig(RVABaseModel):
    model: ModelFile
    score_threshold: float = Field(gt=0, lt=1)
    nms_threshold: float = Field(gt=0, lt=1)
    """IoU above which the weaker of two overlapping detections is dropped."""


class TrackerConfig(RVABaseModel):
    min_face_height: float = Field(gt=0, lt=1)
    """Smaller faces, as a fraction of the frame height, are ignored: they are too far away."""
    iou_threshold: float = Field(gt=0, lt=1)
    """Overlap needed to continue a track from one frame to the next."""
    reid_threshold: float = Field(gt=-1, lt=1)
    """Cosine similarity needed to give a new detection the identity of a track that lost sight of
    its face."""
    confirm_after_s: float = Field(ge=0)
    """How long a face must be seen without a miss before it is announced."""
    leave_after_s: float = Field(gt=0)
    """How long a face must be out of sight before it is said to have left."""
    embed_every_s: float = Field(gt=0)
    embeddings_per_track: int = Field(gt=0)
    """Latest samples kept per track: they are matched, and stored by `remember`."""


class MatchConfig(RVABaseModel):
    threshold: float = Field(gt=-1, lt=1)
    """Cosine similarity needed to recognize someone; 0.363 is OpenCV's value for SFace."""
    margin: float = Field(ge=0)
    """When the two best people score within this of each other, the face counts as unknown."""


class FaceVisionConfig(RVABaseModel):
    device: Literal["cuda", "cpu"]
    """Where the ONNX models run."""
    camera: CameraConfig
    detector: DetectorConfig
    recognizer: ModelFile
    tracker: TrackerConfig
    matching: MatchConfig
    database: Path
    """SQLite file of known people and their face embeddings; `~` is expanded."""
    max_embeddings_per_person: int = Field(gt=0)
    """The oldest samples are dropped beyond this."""
