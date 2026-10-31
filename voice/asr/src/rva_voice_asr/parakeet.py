import asyncio
from typing import Final

import onnx_asr
from onnxruntime import SessionOptions, preload_dlls  # ty: ignore[possibly-missing-import]  # onnxruntime imports it inside a try block

from rva_schemas.audio import AudioClip

SAMPLE_RATE: Final = 16000


class ParakeetRecognizer:
    """`SpeechRecognizer` with NVIDIA Parakeet TDT 0.6B v2 (English) on the GPU.

    The model is read from the Hugging Face cache, and downloaded there on first use.
    """

    def __init__(self) -> None:
        # Load the CUDA 13 libraries from the wheels before any system cuDNN built for CUDA 12.
        preload_dlls()
        options = SessionOptions()
        options.log_severity_level = 3
        self._model: Final = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            sess_options=options,
        )

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
