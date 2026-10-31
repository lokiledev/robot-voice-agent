from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download

from rva_voice_audio.onnx import cpu_session
from rva_voice_audio.vad import SAMPLE_RATE

_WINDOW_SECONDS: Final = 8
_N_FFT: Final = 400
_HOP: Final = 160
_N_MELS: Final = 80


class SmartTurn:
    """Probability that the user finished their turn, from the audio of the turn so far.

    Uses Pipecat's smart-turn v3, which judges intonation and phrasing rather than silence alone.
    The model is read from the Hugging Face cache, and downloaded there on first use.
    """

    def __init__(self) -> None:
        model = hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
        self._session: Final = cpu_session(Path(model))

    def probability(self, audio: npt.NDArray[np.float32]) -> float:
        features = whisper_log_mel(audio)[np.newaxis]
        (output,) = self._session.run(None, {"input_features": features})
        return float(np.asarray(output).reshape(-1)[0])


def whisper_log_mel(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Whisper's input features of the last 8 s of 16 kHz audio, shaped (80, 800).

    Matches Hugging Face's WhisperFeatureExtractor with `do_normalize=True` and padding to 8 s at
    the end, which is how smart-turn was trained.
    """
    window_samples = _WINDOW_SECONDS * SAMPLE_RATE
    window = audio[-window_samples:].astype(np.float64)
    window = (window - window.mean()) / np.sqrt(window.var() + 1e-7)
    window = np.pad(window, (0, window_samples - len(window)))

    padded = np.pad(window, _N_FFT // 2, mode="reflect")
    frame_count = 1 + (len(padded) - _N_FFT) // _HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP][:frame_count]
    power = np.abs(np.fft.rfft(frames * _PERIODIC_HANN, axis=1)) ** 2
    mel = np.maximum(_MEL_FILTERS.T @ power.T, 1e-10)
    log_mel = np.log10(mel)[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0).astype(np.float32)


def _hz_to_mel(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Slaney's scale: linear below 1 kHz, logarithmic above.
    log_step = np.log(6.4) / 27.0
    return np.where(
        hz < 1000.0, 3.0 * hz / 200.0, 15.0 + np.log(np.maximum(hz, 1e-10) / 1000.0) / log_step
    )


def _mel_to_hz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    log_step = np.log(6.4) / 27.0
    return np.where(mel < 15.0, 200.0 * mel / 3.0, 1000.0 * np.exp(log_step * (mel - 15.0)))


def _mel_filters() -> npt.NDArray[np.float64]:
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, _N_FFT // 2 + 1)
    mels = np.linspace(
        _hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(SAMPLE_RATE / 2)), _N_MELS + 2
    )
    filter_freqs = _mel_to_hz(mels)
    slopes = filter_freqs[np.newaxis, :] - fft_freqs[:, np.newaxis]
    widths = np.diff(filter_freqs)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    # Slaney normalization: each filter has the same area.
    return filters * (2.0 / (filter_freqs[2:] - filter_freqs[:-2]))[np.newaxis, :]


_MEL_FILTERS: Final = _mel_filters()
_PERIODIC_HANN: Final = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)
