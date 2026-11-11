from collections import deque
from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel
from rva_voice_audio.config import EndpointConfig
from rva_voice_audio.vad import FRAME_SAMPLES, SAMPLE_RATE

type Audio = npt.NDArray[np.float32]


class SpeechStarted(RVABaseModel):
    pass


class TurnEnded(RVABaseModel, arbitrary_types_allowed=True):
    audio: Audio
    """The whole turn at 16 kHz, from just before speech started."""
    silence_ms: int
    """Silence that ended the turn."""


def _frames(milliseconds: int) -> int:
    return max(1, round(milliseconds * SAMPLE_RATE / 1000 / FRAME_SAMPLES))


class Endpointer:
    """Turns frames and their speech probabilities into turn starts and ends.

    A turn starts after `min_speech_ms` of speech, or `barge_in_min_speech_ms` of clearer speech while
    the robot speaks. It ends when smart-turn, asked after
    `turn_check_silence_ms` of silence, says the user is done, or after `max_silence_ms` of silence,
    or once it is `max_turn_ms` long.
    """

    def __init__(self, config: EndpointConfig, turn_complete: Callable[[Audio], float]) -> None:
        self._config: Final = config
        self._turn_complete: Final = turn_complete
        self._min_speech: Final = _frames(config.min_speech_ms)
        self._barge_in_min_speech: Final = _frames(config.barge_in_min_speech_ms)
        self._turn_check: Final = _frames(config.turn_check_silence_ms)
        self._max_silence: Final = _frames(config.max_silence_ms)
        self._max_turn: Final = _frames(config.max_turn_ms)
        self._pre_roll: Final = _frames(config.pre_roll_ms)
        self._recent: Final = deque[Audio](
            maxlen=self._pre_roll + max(self._min_speech, self._barge_in_min_speech)
        )
        self._turn: list[Audio] | None = None
        self._speech_run = 0
        self._silence_run = 0
        self._checked = False

    def push(
        self, frame: Audio, probability: float, robot_speaking: bool
    ) -> SpeechStarted | TurnEnded | None:
        if self._turn is None:
            return self._wait_for_speech(frame, probability, robot_speaking)
        return self._follow_turn(self._turn, frame, probability)

    def _wait_for_speech(
        self, frame: Audio, probability: float, robot_speaking: bool
    ) -> SpeechStarted | None:
        self._recent.append(frame)
        if robot_speaking:
            threshold, needed = self._config.barge_in_threshold, self._barge_in_min_speech
        else:
            threshold, needed = self._config.speech_threshold, self._min_speech
        self._speech_run = self._speech_run + 1 if probability >= threshold else 0
        if self._speech_run < needed:
            return None
        self._turn = list(self._recent)[-(self._pre_roll + needed) :]
        self._recent.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._checked = False
        return SpeechStarted()

    def _follow_turn(self, turn: list[Audio], frame: Audio, probability: float) -> TurnEnded | None:
        turn.append(frame)
        if probability >= self._config.silence_threshold:
            self._silence_run = 0
            self._checked = False
            return self._end(turn) if len(turn) >= self._max_turn else None
        self._silence_run += 1
        if self._silence_run == self._turn_check and not self._checked:
            self._checked = True
            if self._turn_complete(np.concatenate(turn)) >= self._config.turn_complete_threshold:
                return self._end(turn)
        if self._silence_run >= self._max_silence:
            return self._end(turn)
        return None

    def _end(self, turn: list[Audio]) -> TurnEnded:
        self._turn = None
        silence_ms = self._silence_run * FRAME_SAMPLES * 1000 // SAMPLE_RATE
        return TurnEnded(audio=np.concatenate(turn), silence_ms=silence_ms)
