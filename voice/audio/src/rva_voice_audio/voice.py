import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import Final, assert_never

import numpy as np
import numpy.typing as npt
from loguru import logger
from opentelemetry import trace

from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speech import SpeechRecognizer, SpeechSynthesizer
from rva_schemas.audio import AudioClip
from rva_schemas.dialogue import SpeechEnd, SpeechSpan, Spoken, UserUtterance
from rva_schemas.record import Measurement, RobotSentence, Sound
from rva_voice_audio.devices import Speaker
from rva_voice_audio.echo import RobotSpeech
from rva_voice_audio.endpoint import Endpointer, SpeechStarted, TurnEnded
from rva_voice_audio.pieces import cut
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
        may_interrupt: Callable[[], bool],
        split: Callable[[SpeechSpan], Sequence[SpeechSpan]],
    ) -> None:
        """Listen to a 16 kHz stream.

        Args:
            frames: 512-sample frames of the stream, e.g. `Microphone.frames()`.
            vad: Speech probability of each frame.
            endpointer: Decides when turns start and end.
            recognizer: Transcribes each turn.
            recorder: Receives each frame and its speech probability.
            robot_speech: What the robot is saying, to require clearer speech to interrupt it and
                to drop transcripts of its own voice.
            on_speech_start: Called as soon as the user starts talking, e.g. to stop the robot's
                speech.
            may_interrupt: Asked while the robot speaks whether the speech heard may stop it,
                e.g. only once the person it talks with is seen speaking. Until it says so,
                `on_speech_start` waits; the turn is recognized either way.
            split: Cuts the span of a turn into the pieces that cover it in order, e.g. one per
                speaker when several talked on without a pause; each is an utterance.
        """
        self._frames: Final = frames
        self._vad: Final = vad
        self._endpointer: Final = endpointer
        self._recognizer: Final = recognizer
        self._recorder: Final = recorder
        self._robot_speech: Final = robot_speech
        self._on_speech_start: Final = on_speech_start
        self._may_interrupt: Final = may_interrupt
        self._split: Final = split

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        started_at = datetime.now(UTC)
        # Whether a turn started that may not stop the robot's speech yet.
        held_back = False
        async for frame in self._frames:
            probability = self._vad.probability(frame)
            self._recorder.record(Measurement(name="vad", value=probability))
            self._recorder.record(
                Sound(source="microphone", clip=AudioClip(samples=frame, sample_rate=SAMPLE_RATE))
            )
            match self._endpointer.push(frame, probability, self._robot_speech.speaking):
                case None:
                    if held_back and self._may_interrupt():
                        held_back = False
                        self._on_speech_start()
                case SpeechStarted():
                    started_at = datetime.now(UTC)
                    held_back = self._robot_speech.speaking and not self._may_interrupt()
                    logger.debug("Speech started{}", ", held back" if held_back else "")
                    if not held_back:
                        self._on_speech_start()
                case TurnEnded() as turn:
                    held_back = False
                    ended_at = datetime.now(UTC)
                    # The turn's audio runs up to this frame, the silence that ended it included.
                    heard_at = time.monotonic()
                    speech = SpeechSpan(
                        start=heard_at - len(turn.audio) / SAMPLE_RATE,
                        end=heard_at - turn.silence_ms / 1000,
                    )
                    logger.debug("Turn ended after {} ms of silence", turn.silence_ms)
                    for piece, audio in cut(turn.audio, speech, self._split(speech)):
                        with _tracer.start_as_current_span("asr"):
                            text = await self._recognizer.transcribe(
                                AudioClip(samples=audio, sample_rate=SAMPLE_RATE)
                            )
                        logger.debug(
                            "Heard from {:.2f} to {:.2f}: {!r}", piece.start, piece.end, text
                        )
                        if self._robot_speech.is_echo(text):
                            logger.info("Dropped {!r}: the robot's own voice", text)
                        elif text:
                            yield UserUtterance(
                                text=text,
                                started_at=started_at,
                                ended_at=ended_at,
                                speech=piece,
                                speaker=None,
                            )
                case other:
                    assert_never(other)


class VoiceOutput:
    """`SpeechOutput` to a speaker: synthesizes the next sentence while the current one plays."""

    def __init__(
        self,
        speaker: Speaker,
        synthesizer: SpeechSynthesizer,
        robot_speech: RobotSpeech,
        recorder: Recorder,
    ) -> None:
        """Speak through `speaker`; each sentence is recorded as it starts, each clip once played."""
        self._speaker: Final = speaker
        self._synthesizer: Final = synthesizer
        self._robot_speech: Final = robot_speech
        self._recorder: Final = recorder

    def interrupt(self) -> None:
        """Cut off what the robot is saying; the current `say` returns what was heard."""
        self._speaker.interrupt()

    async def say(self, sentences: AsyncIterator[str], keep_going: Callable[[], bool]) -> Spoken:
        self._speaker.reset()
        # A sentence counts as heard once its last clip has played; None marks that point.
        clips = asyncio.Queue[tuple[str, AudioClip | None] | None](maxsize=4)
        synthesizing = asyncio.create_task(self._synthesize(sentences, clips))
        heard: list[str] = []
        end: SpeechEnd = "finished"
        # Whether the next clip is the first of its sentence.
        starting = True
        try:
            with _tracer.start_as_current_span("speak") as span:
                while (item := await clips.get()) is not None:
                    sentence, clip = item
                    if clip is None:
                        heard.append(sentence)
                        starting = True
                        continue
                    if not heard:
                        span.add_event("first_audio")
                    if starting:
                        # Sentences are synthesized ahead: asked here, as each is about to play.
                        if not keep_going():
                            end = "cut_short"
                            break
                        self._recorder.record(RobotSentence(text=sentence))
                        starting = False
                    self._robot_speech.starts(sentence)
                    played = await self._speaker.play(clip)
                    if played:
                        heard_part = AudioClip(
                            samples=clip.samples[:played], sample_rate=clip.sample_rate
                        )
                        self._recorder.record(Sound(source="robot", clip=heard_part))
                    if played < len(clip.samples):
                        end = "talked_over"
                        break
        finally:
            self._robot_speech.stops()
            synthesizing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await synthesizing
        return Spoken(text=" ".join(heard), end=end)

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
