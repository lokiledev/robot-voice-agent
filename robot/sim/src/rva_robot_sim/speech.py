"""What the simulation's people say, synthesized once and kept, and how their mouths follow it.

The mouths open as loud as their lines are, or the microphone is for a live person.
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
from typing import Final, Self

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import model_validator

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


class LiveMouthConfig(RVABaseModel):
    """How the microphone's level opens a live person's mouth, in dB below full scale."""

    closed_dbfs: float
    """Below this the mouth stays shut: above the room's hum."""
    open_dbfs: float
    """From this it is fully open: someone speaking up close."""

    @model_validator(mode="after")
    def _closed_below_open(self) -> Self:
        if self.closed_dbfs >= self.open_dbfs:
            raise ValueError(
                f"Closed at {self.closed_dbfs} dBFS, not below open at {self.open_dbfs}"
            )
        return self


class SpeechConfig(RVABaseModel):
    """What the people say and hear, and the sound devices it goes through, by name or part of it."""

    directory: Path
    """Where the people's lines are kept once synthesized; `~` is expanded."""
    sample_rate: int
    """Of the room's sound, and so of the synthesized lines: Kokoro's, 24000."""
    room: str
    """Where the scripted lines and the live voice are played: the `rva_sim_room` PipeWire sink of
    config/pipewire/61-rva-sim-room.conf, whose monitor the agent listens to as its microphone."""
    loudspeaker: str | None
    """Where the scripted lines are played too, for whoever watches to hear them; None, and only
    the robot does. The robot's own speaker: its echo canceller then keeps them out of the live
    voice."""
    microphone: str
    """Of whoever speaks for a live person; unused without one."""
    live_mouth: LiveMouthConfig


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


class ScriptedMouth:
    """Opens as loud as their lines are while they say them."""

    def __init__(self, lines: Sequence[SpokenLine]) -> None:
        self._lines: Final = tuple(lines)

    def opening(self, time: float) -> float:
        """How open it is at `time`, in seconds of simulated time, from 0 to 1."""
        return max((line.opening(time) for line in self._lines), default=0.0)


class LiveMouth:
    """Opens as loud as the microphone was in the last moment it heard.

    Told on the audio thread, asked on the simulation's: a float is swapped whole.
    """

    def __init__(self, config: LiveMouthConfig) -> None:
        self._config: Final = config
        self._opening = 0.0

    def hear(self, samples: npt.NDArray[np.float32]) -> None:
        """Open it as loud as `samples`, a few milliseconds of the microphone."""
        rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))
        dbfs = 20 * math.log10(rms) if rms > 0 else -math.inf
        span = self._config.open_dbfs - self._config.closed_dbfs
        self._opening = min(max((dbfs - self._config.closed_dbfs) / span, 0.0), 1.0)

    def opening(self, time: float) -> float:
        """How open it is now, from 0 to 1, whatever `time`: it follows the microphone."""
        return self._opening


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
