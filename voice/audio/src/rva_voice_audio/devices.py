import asyncio
import threading
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt
import sounddevice
from loguru import logger

from rva_schemas.audio import AudioClip
from rva_voice_audio.vad import FRAME_SAMPLES, SAMPLE_RATE


class Microphone:
    """Mono 16 kHz capture in 512-sample frames, the unit Silero VAD scores."""

    def __init__(self, device: str | None) -> None:
        self._frames: Final = asyncio.Queue[npt.NDArray[np.float32]]()
        self._loop: Final = asyncio.get_running_loop()
        self._stream: Final = sounddevice.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._on_audio,
        )

    async def __aenter__(self) -> Self:
        self._stream.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.stop()
        self._stream.close()

    async def frames(self) -> AsyncGenerator[npt.NDArray[np.float32]]:
        while True:
            yield await self._frames.get()

    def _on_audio(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time: object,
        status: sounddevice.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Microphone: {}", status)
        self._loop.call_soon_threadsafe(self._frames.put_nowait, indata[:, 0].copy())


class Speaker:
    """Plays one clip at a time, and can be cut off at any moment."""

    def __init__(self, device: str | None, sample_rate: int) -> None:
        self._sample_rate: Final = sample_rate
        self._loop: Final = asyncio.get_running_loop()
        self._lock: Final = threading.Lock()
        self._clip: npt.NDArray[np.float32] | None = None
        self._position = 0
        self._done: asyncio.Future[bool] | None = None
        self._interrupted = False
        self._stream: Final = sounddevice.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._on_output,
        )

    async def __aenter__(self) -> Self:
        self._stream.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.stop()
        self._stream.close()

    def reset(self) -> None:
        """Allow playing again after an interruption."""
        self._interrupted = False

    def interrupt(self) -> None:
        """Stop the clip playing now and refuse new ones until `reset`."""
        self._interrupted = True
        with self._lock:
            self._clip = None
            done, self._done = self._done, None
        if done is not None and not done.done():
            done.set_result(False)

    async def play(self, clip: AudioClip) -> bool:
        """Play a clip to the end. Returns False if it was interrupted or refused.

        Raises:
            ValueError: If the clip's sample rate is not the speaker's.
        """
        if clip.sample_rate != self._sample_rate:
            raise ValueError(f"Speaker plays {self._sample_rate} Hz, got {clip.sample_rate} Hz")
        if self._interrupted:
            return False
        done = asyncio.Future[bool](loop=self._loop)
        with self._lock:
            self._clip, self._position, self._done = clip.samples, 0, done
        played = await done
        return played

    def _on_output(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: object,
        status: sounddevice.CallbackFlags,
    ) -> None:
        outdata.fill(0)
        with self._lock:
            if self._clip is None:
                return
            chunk = self._clip[self._position : self._position + frames]
            outdata[: len(chunk), 0] = chunk
            self._position += len(chunk)
            if self._position < len(self._clip):
                return
            self._clip = None
            done, self._done = self._done, None
        if done is not None:
            self._loop.call_soon_threadsafe(_finish, done)


def _finish(done: asyncio.Future[bool]) -> None:
    if not done.done():
        done.set_result(True)
