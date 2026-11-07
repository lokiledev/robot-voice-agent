from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from rva_pydantic import RVABaseModel


class ModelFile(RVABaseModel):
    """An ONNX file on the Hugging Face Hub, downloaded on first use."""

    repo: str
    file: str


class TensorRt(RVABaseModel):
    """The GPU through TensorRT in fp16, with CUDA for what it does not take."""

    kind: Literal["tensorrt"] = "tensorrt"
    engine_cache: Path
    """Where the engines are kept, built on first use in a few seconds each and a couple of
    minutes for depth; `~` is expanded."""


class Cuda(RVABaseModel):
    """The GPU through ONNX Runtime's CUDA provider."""

    kind: Literal["cuda"] = "cuda"


class Cpu(RVABaseModel):
    kind: Literal["cpu"] = "cpu"


type Device = Annotated[TensorRt | Cuda | Cpu, Field(discriminator="kind")]


class CameraConfig(RVABaseModel):
    device: Path
    """V4L2 device, e.g. `/dev/video0`."""
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    intrinsics: Path | None
    """YAML file of the camera's `CameraIntrinsics`, at any resolution of the frame's aspect ratio;
    relative to the profile. `None` when not even assumed ones are known."""


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


class EyeContactConfig(RVABaseModel):
    """When the main user's gaze counts as meeting the camera's.

    The gaze's angle off the camera is the median over a short window, so a blink or a bad frame
    does not break eye contact; two thresholds keep it from flickering at the edge.
    """

    window_s: float = Field(gt=0)
    """How much of the gaze's history the median is taken over."""
    enter_deg: float = Field(gt=0)
    """Eye contact starts once the gaze is closer than this to the camera."""
    leave_deg: float = Field(gt=0)
    """Eye contact ends once the gaze is farther than this from the camera."""

    @model_validator(mode="after")
    def _leave_beyond_enter(self) -> "EyeContactConfig":
        if self.leave_deg < self.enter_deg:
            raise ValueError(f"leave_deg {self.leave_deg} is below enter_deg {self.enter_deg}")
        return self


class LipMotionConfig(RVABaseModel):
    """When the main user's lips move enough to look like speech, from the image alone.

    The motion is the standard deviation of the gap between the lips over a short window: it is
    high while the lips open and close with the syllables, and low for a mouth held still, open or
    shut. Two thresholds keep it from flickering at the edge.
    """

    window_s: float = Field(gt=0)
    """How much of the lips' history the motion is taken over; it bridges the pauses between
    words."""
    enter: float = Field(gt=0)
    """The lips start moving once the motion is above this, as a fraction of the mouth's width."""
    leave: float = Field(gt=0)
    """The lips stop moving once the motion is below this."""

    @model_validator(mode="after")
    def _leave_below_enter(self) -> "LipMotionConfig":
        if self.leave > self.enter:
            raise ValueError(f"leave {self.leave} is above enter {self.enter}")
        return self


class FaceFeaturesConfig(RVABaseModel):
    """Models reading eyes, gaze, mouth and expression on each tracked face."""

    landmarks: ModelFile
    """MediaPipe's face mesh v2: 478 points with the irises, on a 256x256 crop."""
    blendshapes: ModelFile
    """MediaPipe's blendshapes: 52 ARKit coefficients from 146 of the mesh points."""
    gaze: ModelFile
    """L2CS-Net trained on Gaze360: gaze yaw and pitch from a 448x448 face crop."""
    expression: ModelFile
    """A ViT fine-tuned on FER2013: seven expressions from a 224x224 face crop."""
    eye_contact: EyeContactConfig
    lip_motion: LipMotionConfig


class BodyConfig(RVABaseModel):
    """Models finding people and their skeletons, hands included."""

    detector: ModelFile
    """YOLOX on COCO; only its person class is used."""
    score_threshold: float = Field(gt=0, lt=1)
    nms_threshold: float = Field(gt=0, lt=1)
    pose: ModelFile
    """RTMW: 133 COCO-WholeBody keypoints, top-down from each person's box."""
    keypoint_threshold: float = Field(gt=0)
    """Keypoints scoring lower are left out."""


class DepthConfig(RVABaseModel):
    """Model estimating metric depth from the camera image alone."""

    model: ModelFile
    """Depth Anything 3 Metric, exported with a fixed square input."""


class VideoRecordingConfig(RVABaseModel):
    """How camera frames are recorded, when recording video is asked for."""

    width: int = Field(gt=0)
    """Frames are scaled down to this width, keeping their aspect ratio."""
    jpeg_quality: int = Field(ge=1, le=100)


class VideoStreamConfig(RVABaseModel):
    """Where the camera image with the tracked faces is published, when streaming is asked for."""

    url: str
    """RTSP URL to publish to, e.g. MediaMTX's `rtsp://127.0.0.1:8554/lune`."""
    width: int = Field(gt=0)
    """Frames are scaled down to this width, keeping their aspect ratio."""


class FaceVisionConfig(RVABaseModel):
    device: Device
    """Where the ONNX models run."""
    camera: CameraConfig
    detector: DetectorConfig
    recognizer: ModelFile
    tracker: TrackerConfig
    matching: MatchConfig
    face_features: FaceFeaturesConfig | None
    """`None` to only detect, track and recognize faces."""
    body: BodyConfig | None
    """`None` to leave out skeletons."""
    depth: DepthConfig | None
    """`None` to leave out depth."""
    database: Path
    """SQLite file of known people and their face embeddings; `~` is expanded."""
    max_embeddings_per_person: int = Field(gt=0)
    """The oldest samples are dropped beyond this."""
    video: VideoRecordingConfig
    stream: VideoStreamConfig
