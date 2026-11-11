import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speaker import SpeakerDetector
from rva_schemas.audio import AudioClip
from rva_schemas.dialogue import SpeechSpan
from rva_voice_asr.parakeet import ParakeetConfig, ParakeetRecognizer
from rva_voice_audio.config import AudioConfig
from rva_voice_audio.devices import Microphone, Speaker
from rva_voice_audio.echo import RobotSpeech
from rva_voice_audio.endpoint import Endpointer
from rva_voice_audio.replay import ReplayedMicrophone, read_sound
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
    split: Callable[[SpeechSpan], Sequence[SpeechSpan]],
    speaker: SpeakerDetector | None,
) -> AsyncGenerator[VoiceInput]:
    recognizer = await _recognizer(asr)
    async with Microphone(
        config.input_device, None if speaker is None else speaker.hear
    ) as microphone:
        yield _voice_input(
            microphone.frames(),
            config,
            recognizer,
            recorder,
            robot_speech,
            on_speech_start,
            may_interrupt,
            split,
        )


async def replayed_voice_input(
    recording: Path,
    started: threading.Event,
    config: AudioConfig,
    asr: ParakeetConfig,
    recorder: Recorder,
    robot_speech: RobotSpeech,
    on_speech_start: Callable[[], None],
    split: Callable[[SpeechSpan], Sequence[SpeechSpan]],
    speaker: SpeakerDetector | None,
) -> VoiceInput:
    """The voice input hearing a recording's sound in place of the microphone.

    The sound starts once `started` is set, and the utterances end a little after it does.
    """
    recognizer = await _recognizer(asr)
    microphone = ReplayedMicrophone(
        await read_sound(recording),
        None if speaker is None else speaker.hear,
        started,
        # Longer than the silence that ends a turn.
        tail_s=2 * config.endpoint.max_silence_ms / 1000,
    )
    return _voice_input(
        microphone.frames(),
        config,
        recognizer,
        recorder,
        robot_speech,
        on_speech_start,
        lambda: True,
        split,
    )


async def _recognizer(asr: ParakeetConfig) -> ParakeetRecognizer:
    recognizer = ParakeetRecognizer(asr)
    # The first run is several times slower, and builds the TensorRT engines the first time;
    # pay it before the first user turn.
    await recognizer.transcribe(
        AudioClip(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    )
    return recognizer


def _voice_input(
    frames: AsyncIterator[npt.NDArray[np.float32]],
    config: AudioConfig,
    recognizer: ParakeetRecognizer,
    recorder: Recorder,
    robot_speech: RobotSpeech,
    on_speech_start: Callable[[], None],
    may_interrupt: Callable[[], bool],
    split: Callable[[SpeechSpan], Sequence[SpeechSpan]],
) -> VoiceInput:
    return VoiceInput(
        frames=frames,
        vad=SileroVad(),
        endpointer=Endpointer(config.endpoint, SmartTurn().probability),
        recognizer=recognizer,
        recorder=recorder,
        robot_speech=robot_speech,
        on_speech_start=on_speech_start,
        may_interrupt=may_interrupt,
        split=split,
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
