import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_voice_audio.vad import FRAME_SAMPLES, SAMPLE_RATE


async def read_sound(path: Path) -> npt.NDArray[np.float32]:
    """The sound of a video or audio file, mono at 16 kHz, decoded by ffmpeg.

    Raises:
        RuntimeError: If ffmpeg cannot decode it, e.g. a video without sound.
    """
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    sound, errors = await process.communicate()
    if process.returncode != 0 or not sound:
        raise RuntimeError(f"ffmpeg found no sound in {path}: {errors.decode().strip()}")
    return np.frombuffer(sound, np.float32)


class ReplayedMicrophone:
    """A recording's sound in 512-sample frames at the pace it was recorded, as a microphone."""

    def __init__(
        self,
        samples: npt.NDArray[np.float32],
        on_frame: Callable[[float, npt.NDArray[np.float32]], None] | None,
        started: threading.Event,
        tail_s: float,
    ) -> None:
        """Replay `samples`, mono at 16 kHz.

        Args:
            samples: The recording.
            on_frame: Also called with each frame and when it was heard, from `time.monotonic`.
            started: The sound starts once this is set, e.g. with the recording's first image.
            tail_s: Silence added after the recording, for its last turn to end.
        """
        self._samples: Final = np.concatenate(
            [samples, np.zeros(round(tail_s * SAMPLE_RATE), np.float32)]
        )
        self._on_frame: Final = on_frame
        self._started: Final = started

    async def frames(self) -> AsyncGenerator[npt.NDArray[np.float32]]:
        """Every whole frame of the recording and its tail, then no more."""
        # Polled: it is set from the camera's thread, where an asyncio.Event cannot be.
        while not self._started.is_set():  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        start = time.monotonic()
        for index in range(len(self._samples) // FRAME_SAMPLES):
            # A frame is heard once its last sample is.
            heard_at = start + (index + 1) * FRAME_SAMPLES / SAMPLE_RATE
            await asyncio.sleep(max(heard_at - time.monotonic(), 0.0))
            frame = self._samples[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
            if self._on_frame is not None:
                self._on_frame(heard_at, frame)
            yield frame
