import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from rva_schemas.record import Record
from rva_voice_asr.parakeet import ParakeetRecognizer
from rva_voice_audio.config import EndpointConfig
from rva_voice_audio.endpoint import Endpointer
from rva_voice_audio.turn import SmartTurn
from rva_voice_audio.vad import SileroVad
from rva_voice_audio.voice import VoiceInput
from rva_voice_tts.kokoro import KokoroConfig, KokoroSynthesizer

KOKORO = Path("~/.cache/robot-voice-agent/kokoro")
ENDPOINT = EndpointConfig(
    speech_threshold=0.5,
    silence_threshold=0.35,
    min_speech_ms=200,
    pre_roll_ms=300,
    turn_check_silence_ms=200,
    max_silence_ms=1500,
    turn_complete_threshold=0.5,
)


class _NoRecorder:
    def record(self, entry: Record) -> None:
        pass


async def _frames(audio: npt.NDArray[np.float32]) -> AsyncGenerator[npt.NDArray[np.float32]]:
    for start in range(0, len(audio) - 511, 512):
        yield audio[start : start + 512]
        await asyncio.sleep(0)


@pytest.mark.download
@pytest.mark.cuda
async def test_synthesized_request_is_heard_back() -> None:
    tts = KokoroSynthesizer(
        KokoroConfig(
            model_path=KOKORO / "kokoro-v1.0.onnx",
            voices_path=KOKORO / "voices-v1.0.bin",
            voice="am_michael",
            speed=1.0,
        )
    )
    (clip,) = [clip async for clip in tts.synthesize("Please put the red cup on the shelf.")]
    at_16k = np.interp(
        np.arange(int(clip.duration * 16000)) / 16000,
        np.arange(len(clip.samples)) / clip.sample_rate,
        clip.samples,
    ).astype(np.float32)
    silence = np.zeros(16000, dtype=np.float32)
    starts: list[None] = []
    voice = VoiceInput(
        frames=_frames(np.concatenate([silence, at_16k, silence, silence])),
        vad=SileroVad(),
        endpointer=Endpointer(ENDPOINT, SmartTurn().probability),
        recognizer=ParakeetRecognizer(),
        recorder=_NoRecorder(),
        on_speech_start=lambda: starts.append(None),
    )

    utterances = [utterance.text async for utterance in voice.utterances()]

    assert utterances == ["Please put the red cup on the shelf."]
    assert len(starts) == 1
