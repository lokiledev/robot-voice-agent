"""The simulated room's sound: the people's lines as they say them, and the live voice."""

import threading
from collections.abc import Sequence
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt
import sounddevice
from loguru import logger

from rva_robot_sim.speech import LiveMouth, SpeechConfig
from rva_schemas.audio import AudioClip

_BLOCK_S: Final = 0.01
"""How much sound each call of the audio threads makes: how often a live mouth moves."""

type _Samples = npt.NDArray[np.float32]


class Mix:
    """Clips playing at once, each from where it is, summed.

    Started on the simulation's thread, played on an audio thread.
    """

    def __init__(self) -> None:
        # Each clip playing and how far into it.
        self._playing: list[tuple[_Samples, int]] = []
        self._lock: Final = threading.Lock()

    def start(self, samples: _Samples) -> None:
        with self._lock:
            self._playing.append((samples, 0))

    def stop(self, samples: _Samples) -> None:
        """Stop the clip started with these very `samples`, if it still plays."""
        with self._lock:
            self._playing[:] = [clip for clip in self._playing if clip[0] is not samples]

    def next(self, frames: int) -> _Samples:
        """The next `frames` samples of all the clips playing, summed, unclipped: zeros if none."""
        mixed = np.zeros(frames, dtype=np.float32)
        with self._lock:
            still: list[tuple[_Samples, int]] = []
            for samples, played in self._playing:
                chunk = samples[played : played + frames]
                mixed[: len(chunk)] += chunk
                if played + frames < len(samples):
                    still.append((samples, played + frames))
            self._playing[:] = still
        return mixed


class Room:
    """Plays the scripted lines as the people say them, and relays the live voice.

    Both go into the room's device, a PipeWire sink whose monitor the agent listens to: the
    simulated robot hears the people as a microphone would. The scripted lines go to the
    loudspeaker too, for whoever watches; the live voice does not, which would echo. The live
    voice is relayed, and opens its person's mouth, once the scenario has begun.
    """

    def __init__(
        self, lines: Sequence[AudioClip], live: LiveMouth | None, config: SpeechConfig
    ) -> None:
        """Open the devices; playing starts on entering.

        Args:
            lines: Every line the scripted people may say.
            live: The mouth of the person the microphone speaks for; None, and it is not opened.
            config: The devices and the sample rate.

        Raises:
            ValueError: If a line has another sample rate than the room's.
            sounddevice.PortAudioError: If a device does not exist.
        """
        rates = {line.sample_rate for line in lines} - {config.sample_rate}
        if rates:
            raise ValueError(f"Lines at {sorted(rates)} Hz, not {config.sample_rate}")
        self._live: Final = live
        # Read on the audio thread: a bool is swapped whole.
        self._begun = False
        self._into_room: Final = Mix()
        self._into_loudspeaker: Final = Mix()
        stream = {
            "samplerate": config.sample_rate,
            "blocksize": round(_BLOCK_S * config.sample_rate),
            "channels": 1,
            "dtype": "float32",
        }
        self._streams: Final[list[sounddevice.OutputStream | sounddevice.Stream]] = []
        if live is None:
            self._streams.append(
                sounddevice.OutputStream(device=config.room, callback=self._play, **stream)
            )
            logger.info("The room plays into {}", _name(self._streams[0].device))
        else:
            relay = sounddevice.Stream(
                device=(config.microphone, config.room), callback=self._relay, **stream
            )
            self._streams.append(relay)
            microphone, room = relay.device
            logger.info(
                "The room plays into {}, with the live voice from {}",
                _name(room),
                _name(microphone),
            )
        if config.loudspeaker is not None and lines:
            aloud = sounddevice.OutputStream(
                device=config.loudspeaker, callback=self._play_aloud, **stream
            )
            self._streams.append(aloud)
            logger.info("The lines are heard from {}", _name(aloud.device))

    def __enter__(self) -> Self:
        for stream in self._streams:
            stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for stream in self._streams:
            stream.stop()
            stream.close()

    def begin(self) -> None:
        """Relay the live voice from now."""
        self._begun = True

    def play(self, line: AudioClip) -> None:
        """Start `line` now."""
        self._into_room.start(line.samples)
        self._into_loudspeaker.start(line.samples)

    def cut(self, line: AudioClip) -> None:
        """Stop `line` if it still plays, as started with `play`."""
        self._into_room.stop(line.samples)
        self._into_loudspeaker.stop(line.samples)

    def _play(
        self, outdata: _Samples, frames: int, time: object, status: sounddevice.CallbackFlags
    ) -> None:
        if status:
            logger.warning("Room: {}", status)
        outdata[:, 0] = np.clip(self._into_room.next(frames), -1.0, 1.0)

    def _relay(
        self,
        indata: _Samples,
        outdata: _Samples,
        frames: int,
        time: object,
        status: sounddevice.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Room and microphone: {}", status)
        assert self._live is not None, "relaying only with a live person"
        if not self._begun:
            outdata[:, 0] = np.clip(self._into_room.next(frames), -1.0, 1.0)
            return
        voice = indata[:, 0]
        self._live.hear(voice)
        outdata[:, 0] = np.clip(self._into_room.next(frames) + voice, -1.0, 1.0)

    def _play_aloud(
        self, outdata: _Samples, frames: int, time: object, status: sounddevice.CallbackFlags
    ) -> None:
        if status:
            logger.warning("Loudspeaker: {}", status)
        outdata[:, 0] = np.clip(self._into_loudspeaker.next(frames), -1.0, 1.0)


def _name(device: int) -> str:
    return str(sounddevice.query_devices(device)["name"])
