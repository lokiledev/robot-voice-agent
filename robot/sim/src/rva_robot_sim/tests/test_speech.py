from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import pytest

from rva_robot_sim.scenes import Line
from rva_robot_sim.speech import SpokenLine, bake
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
