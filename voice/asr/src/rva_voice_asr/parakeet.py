import asyncio
from pathlib import Path
from typing import Final

import onnx_asr
import tensorrt_libs  # noqa: F401  # Loads TensorRT 10's libraries, which ONNX Runtime opens by name.
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.onnx import TensorRtOptions
from onnxruntime import SessionOptions, preload_dlls  # ty: ignore[possibly-missing-import]  # onnxruntime imports it inside a try block
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.audio import AudioClip

SAMPLE_RATE: Final = 16000


class ParakeetConfig(RVABaseModel):
    engine_cache: Path
    """Where the TensorRT engines are kept, built on first use in about a minute; `~` is expanded."""
    max_utterance_s: float = Field(gt=0)
    """Longest utterance the engines are tuned for; longer ones are still transcribed. Shorter
    bounds take less GPU memory, but little below 20 s."""


class ParakeetRecognizer:
    """`SpeechRecognizer` with NVIDIA Parakeet TDT 0.6B v2 (English), in fp16 on TensorRT.

    TensorRT takes about 6 ms per utterance and 1.9 GB, where ONNX Runtime's CUDA provider took
    39 ms and 3.8 GB, with the same transcripts. The model is read from the Hugging Face cache, and
    downloaded there on first use.
    """

    def __init__(self, config: ParakeetConfig) -> None:
        """Load the model.

        Raises:
            RuntimeError: If the encoder could not run on TensorRT.
        """
        # Load the CUDA 13 libraries from the wheels before any system cuDNN built for CUDA 12.
        preload_dlls()
        # One utterance at a time: onnx_asr's default profile, batches of up to 16, keeps a 3 GB
        # workspace. Its profile is class-wide, set before the sessions are made.
        TensorRtOptions.profile_max_shapes = {
            "batch": 1,
            "waveform_len_ms": round(config.max_utterance_s * 1000),
        }
        TensorRtOptions.profile_opt_shapes = {"batch": 1, "waveform_len_ms": 5000}
        options = SessionOptions()
        options.log_severity_level = 3
        # ONNX Runtime does not make it, and without it falls back to CUDA.
        engine_cache = config.engine_cache.expanduser()
        engine_cache.mkdir(parents=True, exist_ok=True)
        tensorrt = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache),
            "trt_timing_cache_enable": True,
        }
        self._model: Final = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            providers=[
                ("TensorrtExecutionProvider", tensorrt),
                # The decoder, which onnx_asr keeps off TensorRT; a pool grown only as needed
                # holds 0.6 GB less.
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            ],
            sess_options=options,
        )
        # ONNX Runtime retries without TensorRT when it fails, and only says so on stderr.
        asr = self._model.asr
        assert isinstance(asr, NemoConformerTdt)
        providers = asr._encoder.get_providers()  # noqa: SLF001  # onnx_asr keeps its sessions private.
        if providers[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"Parakeet's encoder could not run on TensorRT: {providers}")

    async def transcribe(self, clip: AudioClip) -> str:
        """Transcribe one utterance.

        Raises:
            ValueError: If the clip is not sampled at 16 kHz.
        """
        if clip.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Parakeet needs {SAMPLE_RATE} Hz audio, got {clip.sample_rate} Hz")
        text = await asyncio.to_thread(self._model.recognize, clip.samples, sample_rate=SAMPLE_RATE)
        # One waveform in, one text out; a list only comes back for a list of waveforms.
        assert isinstance(text, str)
        return text.strip()
