import numpy as np
import pytest

from rva_schemas.audio import AudioClip
from rva_schemas.record import Sound
from rva_vision_viewer.levels import SoundLevels


def test_a_sound_gives_its_peak_per_frame_from_when_it_started() -> None:
    levels = SoundLevels(history_s=30.0)
    # Two frames of 32 ms at 16 kHz: a quiet one, then a loud one.
    samples = np.concatenate([np.full(512, 0.1, np.float32), np.full(512, -0.8, np.float32)])

    levels.hear(
        Sound(source="robot", clip=AudioClip(samples=samples, sample_rate=16000)), ended=10.0
    )
    levels.speech(0.9, now=10.0)

    series = levels.series()
    assert series["robot"] == (pytest.approx([9.936, 9.968]), pytest.approx([0.1, 0.8]))
    assert series["speech"] == ([10.0], [0.9])
    assert series["microphone"] == ([], [])


def test_levels_older_than_the_history_are_dropped() -> None:
    levels = SoundLevels(history_s=30.0)

    levels.speech(0.2, now=0.0)
    levels.speech(0.7, now=31.0)

    assert levels.series()["speech"] == ([31.0], [0.7])
