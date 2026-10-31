from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import numpy as np

from rva_schemas.abc.recorder import Recorder
from rva_schemas.audio import AudioClip
from rva_voice_asr.parakeet import ParakeetRecognizer
from rva_voice_audio.config import AudioConfig
from rva_voice_audio.devices import Microphone, Speaker
from rva_voice_audio.endpoint import Endpointer
from rva_voice_audio.turn import SmartTurn
from rva_voice_audio.vad import SileroVad
from rva_voice_audio.voice import VoiceInput, VoiceOutput
from rva_voice_tts.kokoro import KokoroConfig, KokoroSynthesizer

_KOKORO_SAMPLE_RATE = 24000


@asynccontextmanager
async def open_voice_input(
    config: AudioConfig, recorder: Recorder, on_speech_start: Callable[[], None]
) -> AsyncGenerator[VoiceInput]:
    recognizer = ParakeetRecognizer()
    # The first CUDA run is several times slower; pay it before the first user turn.
    await recognizer.transcribe(
        AudioClip(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    )
    endpointer = Endpointer(config.endpoint, SmartTurn().probability)
    async with Microphone(config.input_device) as microphone:
        yield VoiceInput(
            frames=microphone.frames(),
            vad=SileroVad(),
            endpointer=endpointer,
            recognizer=recognizer,
            recorder=recorder,
            on_speech_start=on_speech_start,
        )


@asynccontextmanager
async def open_voice_output(audio: AudioConfig, tts: KokoroConfig) -> AsyncGenerator[VoiceOutput]:
    synthesizer = KokoroSynthesizer(tts)
    async for _ in synthesizer.synthesize("Ready."):
        pass
    async with Speaker(audio.output_device, _KOKORO_SAMPLE_RATE) as speaker:
        yield VoiceOutput(speaker, synthesizer)
