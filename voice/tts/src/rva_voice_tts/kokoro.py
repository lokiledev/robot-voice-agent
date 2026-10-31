import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Final

import numpy as np
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, SessionOptions, preload_dlls  # ty: ignore[possibly-missing-import]  # onnxruntime imports these inside a try block
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.audio import AudioClip


class KokoroConfig(RVABaseModel):
    model_path: Path
    """kokoro-v1.0.onnx from the kokoro-onnx releases; `~` is expanded."""
    voices_path: Path
    """voices-v1.0.bin from the kokoro-onnx releases; `~` is expanded."""
    voice: str
    """Voice name, e.g. am_michael or af_heart."""
    speed: float = Field(gt=0)


class KokoroSynthesizer:
    """`SpeechSynthesizer` with Kokoro-82M on the GPU, one clip per sentence at 24 kHz."""

    def __init__(self, config: KokoroConfig) -> None:
        self._config: Final = config
        # Load the CUDA 13 libraries from the wheels before any system cuDNN built for CUDA 12.
        preload_dlls()
        options = SessionOptions()
        options.log_severity_level = 3
        session = InferenceSession(
            config.model_path.expanduser(),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._kokoro: Final = Kokoro.from_session(session, str(config.voices_path.expanduser()))

    async def synthesize(self, text: str) -> AsyncGenerator[AudioClip]:
        samples, sample_rate = await asyncio.to_thread(
            self._kokoro.create, text, voice=self._config.voice, speed=self._config.speed
        )
        yield AudioClip(samples=samples.astype(np.float32), sample_rate=sample_rate)
