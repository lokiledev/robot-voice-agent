from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import model_validator

from rva_agent.config import AgentConfig
from rva_attention.config import VisualAttentionConfig
from rva_body_client.client import HttpBodyClientConfig
from rva_interaction.config import InteractionConfig
from rva_llm.client import OpenAiLanguageModelConfig
from rva_llm.server import LlamaServerConfig
from rva_multimodal.config import ActiveSpeakerConfig
from rva_observability.session import ObservabilityConfig
from rva_pydantic import RVABaseModel, load_yaml
from rva_robot_client.client import RobotClientConfig
from rva_robot_sim.config import ReachyMiniDescriptionConfig
from rva_schemas.profile import Admins
from rva_ui.media import MediaServerConfig
from rva_ui.server import UiConfig
from rva_vision_faces.config import FaceVisionConfig, RobotSource
from rva_vision_viewer.stream import VideoStreamConfig
from rva_vla_client.client import HttpVlaClientConfig
from rva_voice_asr.parakeet import ParakeetConfig
from rva_voice_audio.config import AudioConfig
from rva_voice_tts.kokoro import KokoroConfig


class HeadControllerConfig(RVABaseModel):
    """Reachy Mini's head controller, which puts the head where it is told."""

    description: ReachyMiniDescriptionConfig
    """The MJCF the head's geometry is read from: the simulation's, downloaded on first use."""
    max_head_yaw_deg: float
    """How far the head may turn from the body before the body follows, in degrees. Pollen's
    SDK allows 65, where the simulated head strays by a few degrees."""
    gaze_position: tuple[float, float, float]
    """Where the head frame stays while the head looks at the gaze targets it is sent, in the
    robot's base frame, in meters: only its pitch and yaw change. Pollen's SDK holds it at
    (0, 0, 0.177), which leaves the platform room to turn the head."""


class RuntimeConfig(RVABaseModel):
    """The machine side of a profile: models, endpoints and recording."""

    agent: AgentConfig
    llm: OpenAiLanguageModelConfig
    llm_server: LlamaServerConfig
    vla: HttpVlaClientConfig
    body: HttpBodyClientConfig
    robot: RobotClientConfig | None
    """The robot's joints and head camera, for now of the simulation `lune-sim` runs; `None`
    without a robot body."""
    head_controller: HeadControllerConfig | None
    """`lune head`, which drives the robot's head to the poses it is sent; `None` without one."""
    audio: AudioConfig
    asr: ParakeetConfig
    tts: KokoroConfig
    vision: FaceVisionConfig
    attention: VisualAttentionConfig
    """Which face or hand in view to attend to, from what the camera sees."""
    stream: VideoStreamConfig
    """Where the camera with the overlay is published for the session page, with `--ui`."""
    active_speaker: ActiveSpeakerConfig | None
    """Which face speaks, from the camera and the microphone; `None` to leave it out."""
    interaction: InteractionConfig
    """Who pays attention to the robot, and who each utterance is from and to."""
    admins: Path
    """YAML file of the admins' user ids, outside the source tree like the face database their ids
    refer to; `~` is expanded. Without it there are no admins."""
    ui: UiConfig
    media_server: MediaServerConfig
    observability: ObservabilityConfig

    @model_validator(mode="after")
    def _robot_for_its_camera_and_head(self) -> Self:
        if isinstance(self.vision.camera.source, RobotSource) and self.robot is None:
            raise ValueError("vision.camera.source is the robot's, but there is no robot")
        if self.head_controller is not None and self.robot is None:
            raise ValueError("There is a head controller, but no robot for it to drive")
        return self

    @classmethod
    def load(cls, profile: Path) -> "RuntimeConfig":
        """The profile's runtime config, with the paths it gives relative to it made whole."""
        config = load_yaml(cls, profile / "runtime.yaml")
        camera = config.vision.camera
        if camera.intrinsics is None:
            return config
        camera = camera.model_copy(update={"intrinsics": profile / camera.intrinsics})
        return config.model_copy(
            update={"vision": config.vision.model_copy(update={"camera": camera})}
        )


def load_admins(path: Path) -> Admins:
    """The admins listed in `path`, or none if there is no such file."""
    path = path.expanduser()
    if not path.exists():
        logger.warning(
            "No admins file at {}: with the camera on, nobody can make the robot act. To become an"
            " admin, copy config/admins.example.yaml there and add your user id from `lune people`.",
            path,
        )
        return Admins(user_ids=frozenset())
    return load_yaml(Admins, path)
