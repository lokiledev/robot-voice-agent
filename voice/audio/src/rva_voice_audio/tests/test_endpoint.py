import numpy as np

from rva_voice_audio.config import EndpointConfig
from rva_voice_audio.endpoint import Audio, Endpointer, SpeechStarted, TurnEnded
from rva_voice_audio.turn import whisper_log_mel

# At 32 ms per frame: speech starts after 3 frames, smart-turn is asked after 7 silent frames and
# the turn ends anyway after 47.
CONFIG = EndpointConfig(
    speech_threshold=0.5,
    silence_threshold=0.35,
    min_speech_ms=96,
    pre_roll_ms=64,
    turn_check_silence_ms=224,
    max_silence_ms=1500,
    turn_complete_threshold=0.5,
)


def _frame(value: float) -> Audio:
    return np.full(512, value, dtype=np.float32)


def _run(endpointer: Endpointer, probabilities: list[float]) -> list[tuple[int, str]]:
    """Push a frame per probability, its value being its index, and list (index, event)."""
    events = []
    for index, probability in enumerate(probabilities):
        match endpointer.push(_frame(index), probability):
            case SpeechStarted():
                events.append((index, "started"))
            case TurnEnded() as turn:
                first, last = int(turn.audio[0]), int(turn.audio[-1])
                events.append((index, f"ended frames {first}-{last} after {turn.silence_ms} ms"))
            case None:
                pass
    return events


def test_turn_ends_when_smart_turn_says_it_is_complete() -> None:
    asked: list[int] = []

    def complete(audio: Audio) -> float:
        asked.append(len(audio) // 512)
        return 0.9

    events = _run(Endpointer(CONFIG, complete), [0.1] * 5 + [0.9] * 10 + [0.1] * 10)

    assert events == [(7, "started"), (21, "ended frames 3-21 after 224 ms")]
    assert asked == [19]


def test_incomplete_turn_waits_for_the_silence_limit() -> None:
    events = _run(Endpointer(CONFIG, lambda audio: 0.1), [0.9] * 5 + [0.1] * 50)

    assert events == [(2, "started"), (51, "ended frames 0-51 after 1504 ms")]


def test_speech_resuming_after_a_pause_continues_the_turn() -> None:
    probabilities = [0.9] * 5 + [0.1] * 7 + [0.9] * 3 + [0.1] * 7
    asked = []

    events = _run(Endpointer(CONFIG, lambda audio: asked.append(len(audio)) or 0.1), probabilities)

    assert events == [(2, "started")]
    assert len(asked) == 2


def test_short_noise_does_not_start_a_turn() -> None:
    assert _run(Endpointer(CONFIG, lambda audio: 1.0), [0.9, 0.9, 0.1, 0.9, 0.1]) == []


def test_log_mel_of_silence_is_the_floor() -> None:
    features = whisper_log_mel(np.zeros(16000, dtype=np.float32))

    assert features.shape == (80, 800)
    assert np.all(features == -1.5)
