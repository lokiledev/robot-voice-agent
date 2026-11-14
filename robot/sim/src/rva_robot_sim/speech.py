"""What the simulation's people say: their lines synthesized once and kept, and how loud they are.

The synthesizer is given, by voice: the simulation does not know which one it is.
"""

import asyncio
import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger

from rva_pydantic import RVABaseModel
from rva_robot_sim.scenes import Line
from rva_robot_sim.skinning import Floats
from rva_schemas.abc.speech import SpeechSynthesizer
from rva_schemas.audio import AudioClip

_LOUDNESS_HOP_S: Final = 0.01
"""How long each step of a line's loudness covers."""
_LOUDEST_PERCENTILE: Final = 95
"""The loudness that opens the mouth fully, as a percentile of the line's: a few peaks louder
still do not open it wider."""


class SpeechConfig(RVABaseModel):
    directory: Path
    """Where the people's lines are kept once synthesized; `~` is expanded."""
    sink: str | None
    """Sound device the lines are played into, by name or part of it: the `rva_sim_room` PipeWire
    sink of config/pipewire/61-rva-sim-room.conf, whose monitor the agent listens to. None, and
    they are not played; the mouths still move."""


class SpokenLine(RVABaseModel, arbitrary_types_allowed=True):
    """A line as it sounds, and when it starts."""

    at_s: float
    """When it starts, in seconds of simulated time."""
    clip: AudioClip

    @property
    def end_s(self) -> float:
        return self.at_s + self.clip.duration

    def opening(self, time: float) -> float:
        """How open the mouth is at `time`, from 0 to 1: as loud as the line is then; 0 outside."""
        step = math.floor((time - self.at_s) / _LOUDNESS_HOP_S)
        if not 0 <= step < len(self._loudness):
            return 0.0
        return float(self._loudness[step])

    @cached_property
    def _loudness(self) -> Floats:
        hop = round(_LOUDNESS_HOP_S * self.clip.sample_rate)
        steps = len(self.clip.samples) // hop
        windows = self.clip.samples[: steps * hop].reshape(steps, hop).astype(np.float64)
        rms = np.sqrt(np.mean(windows**2, axis=1))
        loudest = np.percentile(rms, _LOUDEST_PERCENTILE)
        return np.clip(rms / loudest, 0.0, 1.0) if loudest > 0 else np.zeros(steps)


async def bake(
    lines: Sequence[Line],
    voice: str,
    synthesizer: Callable[[str], SpeechSynthesizer],
    directory: Path,
) -> list[SpokenLine]:
    """The lines as `voice` says them, synthesized the first time and kept in `directory`.

    Args:
        lines: What is said and when.
        voice: Whose voice says them, as the synthesizer names it.
        synthesizer: Gives the synthesizer of a voice; asked only for lines not baked yet.
        directory: Where lines are kept, by voice and text: one per synthesizer and its
            settings, since what is kept does not say which said it.
    """
    spoken = []
    for line in lines:
        path = _path(directory, voice, line.text)
        if not await asyncio.to_thread(path.exists):
            logger.info("Synthesizing {!r} in {}'s voice", line.text, voice)
            clips = [clip async for clip in synthesizer(voice).synthesize(line.text)]
            rates = {clip.sample_rate for clip in clips}
            assert len(rates) == 1, f"one sample rate, not {rates}"
            samples = np.concatenate([clip.samples for clip in clips])
            await asyncio.to_thread(_keep, path, samples, rates.pop())
        clip = await asyncio.to_thread(_load, path)
        spoken.append(SpokenLine(at_s=line.at_s, clip=clip))
    return spoken


def _path(directory: Path, voice: str, text: str) -> Path:
    return directory.expanduser() / voice / f"{hashlib.sha256(text.encode()).hexdigest()[:24]}.npz"


def _load(path: Path) -> AudioClip:
    with np.load(path) as data:
        return AudioClip(samples=data["samples"], sample_rate=int(data["sample_rate"]))


def _keep(path: Path, samples: Floats, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Moved in place once whole: an interrupted run leaves no half line.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        np.savez(file, samples=samples.astype(np.float32), sample_rate=sample_rate)
    os.replace(file.name, path)
