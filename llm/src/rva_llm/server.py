from pathlib import Path

from huggingface_hub import hf_hub_download
from pydantic import Field

from rva_pydantic import RVABaseModel


class LlamaServerConfig(RVABaseModel):
    """How to launch llama.cpp's llama-server for the agent's LLM."""

    executable: Path
    hf_repo: str
    """Hugging Face repository of the GGUF file, e.g. unsloth/Qwen3.5-35B-A3B-GGUF."""
    hf_file: str
    """GGUF file in the repository, e.g. Qwen3.5-35B-A3B-Q4_K_M.gguf."""
    host: str
    port: int
    context_size: int = Field(gt=0)
    """In tokens."""
    n_cpu_moe: int = Field(ge=0)
    """Layers whose MoE experts stay in system RAM, to fit the model beside ASR and TTS."""
    extra_args: tuple[str, ...]
    """Further llama-server flags."""


def llama_server_command(config: LlamaServerConfig) -> list[str]:
    """The llama-server command line for `config`.

    Raises:
        huggingface_hub.errors.LocalEntryNotFoundError: If the GGUF file has not been downloaded.
    """
    model = hf_hub_download(config.hf_repo, config.hf_file, local_files_only=True)
    return [
        str(config.executable.expanduser()),
        "--model", model,
        "--host", config.host,
        "--port", str(config.port),
        "--ctx-size", str(config.context_size),
        "--n-gpu-layers", "999",
        "--n-cpu-moe", str(config.n_cpu_moe),
        "--jinja",
        *config.extra_args,
    ]  # fmt: skip
