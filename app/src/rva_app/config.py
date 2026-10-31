from pathlib import Path

from loguru import logger

from rva_agent.config import AgentConfig
from rva_body_client.client import HttpBodyClientConfig
from rva_llm.client import OpenAiLanguageModelConfig
from rva_llm.server import LlamaServerConfig
from rva_observability.session import ObservabilityConfig
from rva_pydantic import RVABaseModel, load_yaml
from rva_schemas.profile import Admins
from rva_ui.media import MediaServerConfig
from rva_ui.server import UiConfig
from rva_vision_faces.config import FaceVisionConfig
from rva_vla_client.client import HttpVlaClientConfig
from rva_voice_audio.config import AudioConfig
from rva_voice_tts.kokoro import KokoroConfig


class RuntimeConfig(RVABaseModel):
    """The machine side of a profile: models, endpoints and recording."""

    agent: AgentConfig
    llm: OpenAiLanguageModelConfig
    llm_server: LlamaServerConfig
    vla: HttpVlaClientConfig
    body: HttpBodyClientConfig
    audio: AudioConfig
    tts: KokoroConfig
    vision: FaceVisionConfig
    admins: Path
    """YAML file of the admins' user ids, outside the source tree like the face database their ids
    refer to; `~` is expanded. Without it there are no admins."""
    ui: UiConfig
    media_server: MediaServerConfig
    observability: ObservabilityConfig

    @classmethod
    def load(cls, profile: Path) -> "RuntimeConfig":
        return load_yaml(cls, profile / "runtime.yaml")


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
