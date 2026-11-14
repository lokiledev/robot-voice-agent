from pathlib import Path
from typing import Final, Self

import yaml
from loguru import logger
from pydantic import Field, JsonValue, TypeAdapter, model_validator

from rva_agent.config import AgentConfig
from rva_app.doa import DoaConfig, RobotDoaConfig
from rva_attention.config import VisualAttentionConfig
from rva_body_client.client import HttpBodyClientConfig
from rva_interaction.config import InteractionConfig
from rva_llm.client import OpenAiLanguageModelConfig
from rva_llm.server import LlamaServerConfig
from rva_multimodal.config import ActiveSpeakerConfig
from rva_observability.session import ObservabilityConfig
from rva_pydantic import RVABaseModel, load_yaml
from rva_robot_client.client import RobotClientConfig
from rva_robot_kinematics.motion import MotionLimits
from rva_robot_sim.config import ReachyMiniDescriptionConfig
from rva_schemas.frames import FrameTransform, StaticTransforms
from rva_schemas.profile import Admins
from rva_ui.media import MediaServerConfig
from rva_ui.server import UiConfig
from rva_vision_faces.config import FaceVisionConfig, RobotSource
from rva_vision_viewer.stream import VideoStreamConfig
from rva_vla_client.client import HttpVlaClientConfig
from rva_voice_asr.parakeet import ParakeetConfig
from rva_voice_audio.config import AudioConfig
from rva_voice_tts.kokoro import KokoroConfig

_EXTENDS: Final = "extends"
"""The key of a `runtime.yaml` that names the directory of the profile it is laid over, relative
to its own."""
_KIND: Final = "kind"
"""The tag that tells the variants of a setting apart."""
_SETTINGS: Final = TypeAdapter(dict[str, JsonValue])


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
    motion: MotionLimits
    """How fast and how abruptly the head moves towards where it is sent."""
    recentre_after_s: float = Field(gt=0)
    """How long after the last gaze target the head drifts back to looking ahead."""
    recentre_speed_deg_s: float = Field(gt=0)
    """How fast it drifts back."""


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
    doa: DoaConfig | None
    """The microphone array that tells where speech comes from, its direction of arrival, to
    tell who said it: the reSpeaker, or the robot's own; `None` without one."""
    transforms: tuple[FrameTransform, ...]
    """Where links are fixed relative to each other, until the robot's description tells it."""
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
        if self.doa is not None:
            if self.active_speaker is None:
                raise ValueError("Directions of arrival only help the active speaker")
            if isinstance(self.doa, RobotDoaConfig) and self.robot is None:
                raise ValueError("doa is the robot's, but there is no robot")
            camera, array = self.vision.camera.frame, self.doa.frame
            if not StaticTransforms(self.transforms).connects(camera, array):
                raise ValueError(f"No transform between the camera's {camera} and the {array}")
        return self

    @classmethod
    def load(cls, profile: Path) -> "RuntimeConfig":
        """The profile's runtime config, with the paths it gives relative to it made whole.

        A `runtime.yaml` that names another profile's directory under `extends` only holds what
        differs from that profile's, and is laid over it: see `_laid_over`.

        Raises:
            FileNotFoundError: Without a `runtime.yaml` in `profile`, or in a profile it extends.
            pydantic.ValidationError: If the settings, once laid over each other, do not fit.
        """
        return cls.model_validate(_settings(profile))


def _settings(profile: Path) -> dict[str, JsonValue]:
    """What the profile's `runtime.yaml` holds, over that of the profile it extends, if any."""
    path = profile / "runtime.yaml"
    own = _SETTINGS.validate_python(yaml.safe_load(path.read_text()))
    # Relative to the profile that names it, which may not be the one loaded.
    match own:
        case {"vision": {"camera": {"intrinsics": str(intrinsics)}}}:
            whole: JsonValue = {"camera": {"intrinsics": str(profile / intrinsics)}}
            own = _laid_over(own, {"vision": whole})
    match own.pop(_EXTENDS, None):
        case None:
            return own
        case str(base):
            return _laid_over(_settings(profile / base), own)
        case other:
            raise ValueError(f"{path}: {_EXTENDS} is a profile's directory, not {other!r}")


def _laid_over(base: dict[str, JsonValue], override: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """`override` over `base`: what it says replaces what `base` says, the rest is `base`'s.

    Mappings are laid over each other key by key, all the way down, so a profile changes one
    threshold without saying the others again. A list, a value or a null replaces whatever was
    there. So does a mapping of another `kind` than the one it lies over: the fields of one kind
    of camera or of microphone array mean nothing to another.
    """
    if _KIND in base and _KIND in override and base[_KIND] != override[_KIND]:
        return override
    laid = dict(base)
    for key, value in override.items():
        under = base.get(key)
        laid[key] = (
            _laid_over(under, value)
            if isinstance(under, dict) and isinstance(value, dict)
            else value
        )
    return laid


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
