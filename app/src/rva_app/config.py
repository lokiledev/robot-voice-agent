from pathlib import Path

from rva_agent.config import AgentConfig
from rva_body_client.client import HttpBodyClientConfig
from rva_llm.client import OpenAiLanguageModelConfig
from rva_llm.server import LlamaServerConfig
from rva_observability.session import ObservabilityConfig
from rva_pydantic import RVABaseModel, load_yaml
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
    ui: UiConfig
    media_server: MediaServerConfig
    observability: ObservabilityConfig

    @classmethod
    def load(cls, profile: Path) -> "RuntimeConfig":
        return load_yaml(cls, profile / "runtime.yaml")
