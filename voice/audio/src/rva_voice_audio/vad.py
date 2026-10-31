from importlib.resources import files
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_voice_audio.onnx import cpu_session

SAMPLE_RATE: Final = 16000
FRAME_SAMPLES: Final = 512
"""Silero VAD scores 32 ms windows at 16 kHz."""
_CONTEXT_SAMPLES: Final = 64
_MODEL: Final = Path(str(files("rva_voice_audio").joinpath("assets", "silero_vad.onnx")))


class SileroVad:
    """Speech probability of consecutive frames of one 16 kHz stream, with Silero VAD v6."""

    def __init__(self) -> None:
        self._session: Final = cpu_session(_MODEL)
        self._sample_rate: Final = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: npt.NDArray[np.float32]) -> float:
        """Speech probability of the next frame of the stream.

        Raises:
            ValueError: If the frame is not 512 samples long.
        """
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"Expected {FRAME_SAMPLES} samples, got shape {frame.shape}")
        # The model expects the end of the previous frame before the new one.
        model_input = np.concatenate([self._context, frame[np.newaxis, :]], axis=1)
        probability, self._state = self._session.run(
            None, {"input": model_input, "state": self._state, "sr": self._sample_rate}
        )
        self._context = model_input[:, -_CONTEXT_SAMPLES:]
        return float(np.asarray(probability).reshape(-1)[0])
