import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Final, assert_never

import numpy as np
import numpy.typing as npt
from loguru import logger
from opentelemetry import trace

from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speech import SpeechRecognizer, SpeechSynthesizer
from rva_schemas.audio import AudioClip
from rva_schemas.dialogue import Spoken, UserUtterance
from rva_schemas.record import Measurement
from rva_voice_audio.devices import Speaker
from rva_voice_audio.echo import RobotSpeech
from rva_voice_audio.endpoint import Endpointer, SpeechStarted, TurnEnded
from rva_voice_audio.vad import SAMPLE_RATE, SileroVad

_tracer: Final = trace.get_tracer("rva_voice_audio")


class VoiceInput:
    """`UserInput` from a microphone: voice activity, end of turn, then speech recognition."""

    def __init__(
        self,
        frames: AsyncIterator[npt.NDArray[np.float32]],
        vad: SileroVad,
        endpointer: Endpointer,
        recognizer: SpeechRecognizer,
        recorder: Recorder,
        robot_speech: RobotSpeech,
        on_speech_start: Callable[[], None],
    ) -> None:
        """Listen to a 16 kHz stream.

        Args:
            frames: 512-sample frames of the stream, e.g. `Microphone.frames()`.
            vad: Speech probability of each frame.
            endpointer: Decides when turns start and end.
            recognizer: Transcribes each turn.
            recorder: Receives the speech probability of each frame.
            robot_speech: What the robot is saying, to require clearer speech to interrupt it and
                to drop transcripts of its own voice.
            on_speech_start: Called as soon as the user starts talking, e.g. to stop the robot's
                speech.
        """
        self._frames: Final = frames
        self._vad: Final = vad
        self._endpointer: Final = endpointer
        self._recognizer: Final = recognizer
        self._recorder: Final = recorder
        self._robot_speech: Final = robot_speech
        self._on_speech_start: Final = on_speech_start

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        started_at = datetime.now(UTC)
        async for frame in self._frames:
            probability = self._vad.probability(frame)
            self._recorder.record(Measurement(name="vad", value=probability))
            match self._endpointer.push(frame, probability, self._robot_speech.speaking):
                case None:
                    pass
                case SpeechStarted():
                    started_at = datetime.now(UTC)
                    logger.debug("Speech started")
                    self._on_speech_start()
                case TurnEnded() as turn:
                    ended_at = datetime.now(UTC)
                    with _tracer.start_as_current_span("asr"):
                        text = await self._recognizer.transcribe(
                            AudioClip(samples=turn.audio, sample_rate=SAMPLE_RATE)
                        )
                    logger.debug("Turn ended after {} ms of silence: {!r}", turn.silence_ms, text)
                    if self._robot_speech.is_echo(text):
                        logger.info("Dropped {!r}: the robot's own voice", text)
                    elif text:
                        yield UserUtterance(text=text, started_at=started_at, ended_at=ended_at)
                case other:
                    assert_never(other)


class VoiceOutput:
    """`SpeechOutput` to a speaker: synthesizes the next sentence while the current one plays."""

    def __init__(
        self, speaker: Speaker, synthesizer: SpeechSynthesizer, robot_speech: RobotSpeech
    ) -> None:
        self._speaker: Final = speaker
        self._synthesizer: Final = synthesizer
        self._robot_speech: Final = robot_speech

    def interrupt(self) -> None:
        """Cut off what the robot is saying; the current `say` returns what was heard."""
        self._speaker.interrupt()

    async def say(self, sentences: AsyncIterator[str]) -> Spoken:
        self._speaker.reset()
        # A sentence counts as heard once its last clip has played; None marks that point.
        clips = asyncio.Queue[tuple[str, AudioClip | None] | None](maxsize=4)
        synthesizing = asyncio.create_task(self._synthesize(sentences, clips))
        heard: list[str] = []
        interrupted = False
        try:
            with _tracer.start_as_current_span("speak") as span:
                while (item := await clips.get()) is not None:
                    sentence, clip = item
                    if clip is None:
                        heard.append(sentence)
                        continue
                    if not heard:
                        span.add_event("first_audio")
                    self._robot_speech.starts(sentence)
                    if not await self._speaker.play(clip):
                        interrupted = True
                        break
        finally:
            self._robot_speech.stops()
            synthesizing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await synthesizing
        return Spoken(text=" ".join(heard), interrupted=interrupted)

    async def _synthesize(
        self,
        sentences: AsyncIterator[str],
        clips: asyncio.Queue[tuple[str, AudioClip | None] | None],
    ) -> None:
        async for sentence in sentences:
            async for clip in self._synthesizer.synthesize(sentence):
                await clips.put((sentence, clip))
            await clips.put((sentence, None))
        await clips.put(None)
