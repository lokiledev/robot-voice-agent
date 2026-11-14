from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rva_robot_sim.scenes import Line
from rva_robot_sim.speech import LiveMouth, LiveMouthConfig, ScriptedMouth, SpokenLine, bake
from rva_schemas.abc.speech import SpeechSynthesizer
from rva_schemas.audio import AudioClip


class _Synthesizer:
    """Says each text as two clips of a tone, each a tenth of a second per character."""

    def __init__(self, voice: str, said: list[tuple[str, str]]) -> None:
        self._voice = voice
        self._said = said

    async def synthesize(self, text: str) -> AsyncGenerator[AudioClip]:
        self._said.append((self._voice, text))
        for _ in range(2):
            samples = np.full(len(text) * 100, 0.25, np.float32)
            yield AudioClip(samples=samples, sample_rate=1000)


def _refuse(voice: str) -> SpeechSynthesizer:
    raise AssertionError(f"{voice} was asked to synthesize a line kept already")


async def test_lines_are_synthesized_once_then_kept(tmp_path: Path) -> None:
    lines = (Line(at_s=1.0, text="Hi"), Line(at_s=4.0, text="Bye!"))
    said: list[tuple[str, str]] = []

    first = await bake(lines, "af_bella", lambda voice: _Synthesizer(voice, said), tmp_path)
    again = await bake(lines, "af_bella", _refuse, tmp_path)

    assert said == [("af_bella", "Hi"), ("af_bella", "Bye!")]
    for spoken in (first, again):
        assert [line.at_s for line in spoken] == [1.0, 4.0]
        # Both clips of each text, joined.
        assert [line.clip.duration for line in spoken] == [0.4, 0.8]
        assert [line.end_s for line in spoken] == [1.4, 4.8]


async def test_each_voice_keeps_its_own(tmp_path: Path) -> None:
    lines = (Line(at_s=0.0, text="Hi"),)
    said: list[tuple[str, str]] = []

    await bake(lines, "af_bella", lambda voice: _Synthesizer(voice, said), tmp_path)
    await bake(lines, "am_michael", lambda voice: _Synthesizer(voice, said), tmp_path)

    assert said == [("af_bella", "Hi"), ("am_michael", "Hi")]


@pytest.mark.parametrize(
    ("time", "opening"),
    [
        (0.5, 0.0),  # Before it starts.
        (1.05, 0.0),  # Its silent first half.
        (1.75, 1.0),  # Its loud second half: as loud as it gets.
        (2.5, 0.0),  # After it ends.
    ],
)
def test_the_mouth_opens_as_loud_as_the_line(time: float, opening: float) -> None:
    samples = np.concatenate([np.zeros(500), np.full(500, 0.3)]).astype(np.float32)
    line = SpokenLine(at_s=1.0, clip=AudioClip(samples=samples, sample_rate=1000))

    assert line.opening(time) == pytest.approx(opening)


def test_a_scripted_mouth_opens_for_the_line_said() -> None:
    loud = AudioClip(samples=np.full(1000, 0.3, np.float32), sample_rate=1000)
    mouth = ScriptedMouth([SpokenLine(at_s=1.0, clip=loud), SpokenLine(at_s=5.0, clip=loud)])

    assert [mouth.opening(time) for time in (0.5, 1.5, 3.0, 5.5)] == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("level", "opening"),
    [
        (0.0, 0.0),  # Silence.
        (0.001, 0.0),  # The room's hum: -60 dBFS.
        (10 ** (-40 / 20), 0.5),  # Halfway from closed to open, in dB.
        (0.5, 1.0),  # Shouting: -6 dBFS.
    ],
)
def test_a_live_mouth_opens_as_loud_as_the_microphone(level: float, opening: float) -> None:
    mouth = LiveMouth(LiveMouthConfig(closed_dbfs=-50, open_dbfs=-30))

    mouth.hear(np.full(240, level, np.float32))

    assert mouth.opening(123.0) == pytest.approx(opening)


def test_a_live_mouth_closes_below_where_it_opens() -> None:
    with pytest.raises(ValidationError, match="not below open"):
        LiveMouthConfig(closed_dbfs=-20, open_dbfs=-30)
