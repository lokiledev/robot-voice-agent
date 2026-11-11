import numpy as np
import pytest

from rva_schemas.dialogue import SpeechSpan
from rva_voice_audio.pieces import cut


def test_a_turn_is_cut_at_the_quietest_moment_near_where_the_speaker_changes() -> None:
    # Three seconds of sound from t = 10 s, silent for 20 ms from 1.11 s in.
    audio = np.ones(48000, np.float32)
    audio[17760:18080] = 0.0
    speech = SpeechSpan(start=10.0, end=12.5)

    (first, first_audio), (second, second_audio) = cut(
        audio, speech, [SpeechSpan(start=10.0, end=11.0), SpeechSpan(start=11.0, end=12.5)]
    )

    # Asked at 1.0 s, the cut moves to the middle of the silence, 1.12 s in.
    assert (len(first_audio), len(second_audio)) == (17920, 30080)
    assert (first.start, first.end) == (10.0, pytest.approx(11.12))
    assert (second.start, second.end) == (pytest.approx(11.12), 12.5)


def test_a_turn_in_one_piece_is_left_whole() -> None:
    audio = np.ones(16000, np.float32)
    speech = SpeechSpan(start=0.0, end=0.8)

    pieces = cut(audio, speech, [speech])

    assert [(span, len(piece)) for span, piece in pieces] == [(speech, 16000)]
