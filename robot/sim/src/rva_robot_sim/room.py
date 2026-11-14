"""The simulated room's sound: the people's lines, played into a sound device as they come due."""

import threading
from collections.abc import Sequence
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt
import sounddevice
from loguru import logger

from rva_robot_sim.speech import SpokenLine


class Room:
    """Plays the people's lines into a sound device when the simulation reaches them, mixed.

    The device is a PipeWire sink the agent listens to through its monitor: the simulated robot
    hears the people as a microphone would, with the room's other sounds mixed in by PipeWire.
    """

    def __init__(self, lines: Sequence[SpokenLine], device: str, sample_rate: int) -> None:
        """Open the device; playing starts on entering.

        Args:
            lines: Everyone's lines.
            device: The sound device, by name or part of it.
            sample_rate: Of every line, in Hz.

        Raises:
            ValueError: If a line has another sample rate, or there is no such device.
        """
        rates = {line.clip.sample_rate for line in lines} - {sample_rate}
        if rates:
            raise ValueError(f"Lines at {sorted(rates)} Hz, not {sample_rate}")
        self._due: Final = sorted(lines, key=lambda line: line.at_s)
        self._next = 0
        # Lines playing and how far into each, shared with the audio thread.
        self._playing: list[tuple[npt.NDArray[np.float32], int]] = []
        self._lock: Final = threading.Lock()
        self._stream: Final = sounddevice.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._on_output,
        )
        logger.info(
            "The people speak into {}", sounddevice.query_devices(self._stream.device)["name"]
        )

    def __enter__(self) -> Self:
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.stop()
        self._stream.close()

    def speak(self, simulated: float) -> None:
        """Start the lines due by `simulated` seconds, which have not started yet."""
        while self._next < len(self._due) and self._due[self._next].at_s <= simulated:
            line = self._due[self._next]
            self._next += 1
            with self._lock:
                self._playing.append((line.clip.samples, 0))

    def _on_output(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        stream_time: object,
        status: sounddevice.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Room: {}", status)
        mixed = np.zeros(frames, dtype=np.float32)
        with self._lock:
            still: list[tuple[npt.NDArray[np.float32], int]] = []
            for samples, played in self._playing:
                chunk = samples[played : played + frames]
                mixed[: len(chunk)] += chunk
                if played + frames < len(samples):
                    still.append((samples, played + frames))
            self._playing[:] = still
        outdata[:, 0] = np.clip(mixed, -1.0, 1.0)
