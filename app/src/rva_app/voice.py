import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import numpy as np

from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speaker import SpeakerDetector
from rva_schemas.audio import AudioClip
from rva_voice_asr.parakeet import ParakeetConfig, ParakeetRecognizer
from rva_voice_audio.config import AudioConfig
from rva_voice_audio.devices import Microphone, Speaker
from rva_voice_audio.echo import RobotSpeech
from rva_voice_audio.endpoint import Endpointer
from rva_voice_audio.turn import SmartTurn
from rva_voice_audio.vad import SileroVad
from rva_voice_audio.voice import VoiceInput, VoiceOutput
from rva_voice_tts.kokoro import KokoroConfig, KokoroSynthesizer

_KOKORO_SAMPLE_RATE = 24000


@asynccontextmanager
async def open_voice_input(
    config: AudioConfig,
    asr: ParakeetConfig,
    recorder: Recorder,
    robot_speech: RobotSpeech,
    on_speech_start: Callable[[], None],
    may_interrupt: Callable[[], bool],
    speaker: SpeakerDetector | None,
) -> AsyncGenerator[VoiceInput]:
    recognizer = ParakeetRecognizer(asr)
    # The first run is several times slower, and builds the TensorRT engines the first time;
    # pay it before the first user turn.
    await recognizer.transcribe(
        AudioClip(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    )
    endpointer = Endpointer(config.endpoint, SmartTurn().probability)
    async with Microphone(
        config.input_device, None if speaker is None else speaker.hear
    ) as microphone:
        yield VoiceInput(
            frames=microphone.frames(),
            vad=SileroVad(),
            endpointer=endpointer,
            recognizer=recognizer,
            recorder=recorder,
            robot_speech=robot_speech,
            on_speech_start=on_speech_start,
            may_interrupt=may_interrupt,
        )


@asynccontextmanager
async def open_voice_output(
    audio: AudioConfig, tts: KokoroConfig, robot_speech: RobotSpeech, recorder: Recorder
) -> AsyncGenerator[VoiceOutput]:
    synthesizer = KokoroSynthesizer(tts)
    async for _ in synthesizer.synthesize("Ready."):
        pass
    async with Speaker(audio.output_device, _KOKORO_SAMPLE_RATE) as speaker:
        yield VoiceOutput(speaker, synthesizer, robot_speech, recorder)


def robot_speech(config: AudioConfig) -> RobotSpeech:
    return RobotSpeech(config.echo_window_ms / 1000, time.monotonic)
