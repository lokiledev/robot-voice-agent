import threading
from collections import deque
from typing import Final

import numpy as np

from rva_schemas.record import Sound

_HOP_S: Final = 0.032
"""How long each point of a sound's level covers: a frame of the microphone."""
SPEECH: Final = "speech"
"""The series of how likely the microphone heard speech, beside one per source of sound."""


class SoundLevels:
    """How loud the microphone and the robot's voice were lately, and how likely speech was heard.

    Fed from any thread, on one clock, e.g. `time.monotonic`.
    """

    def __init__(self, history_s: float) -> None:
        self._history_s: Final = history_s
        self._series: Final[dict[str, deque[tuple[float, float]]]] = {
            "microphone": deque(),
            "robot": deque(),
            SPEECH: deque(),
        }
        self._lock: Final = threading.Lock()

    def hear(self, sound: Sound, ended: float) -> None:
        """Add the level of a sound that ended at `ended`, a point per frame of it."""
        clip = sound.clip
        hop = round(_HOP_S * clip.sample_rate)
        start = ended - clip.duration
        points = [
            (
                start + offset / clip.sample_rate,
                float(np.abs(clip.samples[offset : offset + hop]).max()),
            )
            for offset in range(0, len(clip.samples), hop)
        ]
        self._add(sound.source, points, ended)

    def speech(self, probability: float, now: float) -> None:
        self._add(SPEECH, [(now, probability)], now)

    def series(self) -> dict[str, tuple[list[float], list[float]]]:
        """The times and the values of each series, oldest first; from 0 to 1."""
        with self._lock:
            return {
                name: ([time for time, _ in points], [value for _, value in points])
                for name, points in self._series.items()
            }

    def _add(self, name: str, points: list[tuple[float, float]], now: float) -> None:
        with self._lock:
            series = self._series[name]
            series.extend(points)
            while series and series[0][0] < now - self._history_s:
                series.popleft()
