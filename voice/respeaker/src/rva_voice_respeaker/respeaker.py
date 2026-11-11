import asyncio
import math
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from loguru import logger

from rva_pydantic import RVABaseModel
from rva_schemas.abc.recorder import Recorder
from rva_schemas.record import Measurement
from rva_schemas.sound import HeardDirection
from rva_voice_respeaker.config import ReSpeakerConfig
from rva_voice_respeaker.control import BEAM_AZIMUTHS, BEAM_SPEECH_ENERGIES, XvfControl

BEAMS: Final = ("focused_1", "focused_2", "scanning", "selected")
"""The chip's beams, in the order it reports them."""


class BoardReading(RVABaseModel):
    """What the array reported at one moment, in its own frame."""

    time: float
    """When the sound was heard, on the monotonic clock."""
    speech_azimuth: float | None
    """Direction of the speaker in radians, from `speaker_azimuth`; `None` when it hears no
    speech."""
    beam_azimuths: tuple[float, float, float, float]
    """Of each of `BEAMS`, in radians."""
    beam_speech_energies: tuple[float, float, float, float]
    """Of each of `BEAMS`: above 0 is speech, the louder or closer the higher."""


class ReSpeaker:
    """Asks the reSpeaker XVF3800 where it hears speech from, at a steady pace."""

    def __init__(self, config: ReSpeakerConfig, control: XvfControl, recorder: Recorder) -> None:
        self._config: Final = config
        self._control: Final = control
        self._recorder: Final = recorder

    async def readings(self) -> AsyncGenerator[BoardReading]:
        """Each reading as it comes, recorded too, until cancelled."""
        period = 1 / self._config.poll_hz
        while True:
            started = time.monotonic()
            reading = await asyncio.to_thread(self._read)
            self._record(reading)
            yield reading
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def directions(self) -> AsyncGenerator[HeardDirection]:
        """Where speech is heard from, as the array's frame has it, until cancelled."""
        async for reading in self.readings():
            yield HeardDirection(
                time=reading.time, frame=self._config.frame, speech_azimuth=reading.speech_azimuth
            )

    def _read(self) -> BoardReading:
        heard = time.monotonic() - self._config.delay_s
        azimuths = self._control.read(BEAM_AZIMUTHS)
        energies = self._control.read(BEAM_SPEECH_ENERGIES)
        beam_azimuths = (azimuths[0], azimuths[1], azimuths[2], azimuths[3])
        beam_energies = (energies[0], energies[1], energies[2], energies[3])
        return BoardReading(
            time=heard,
            speech_azimuth=speaker_azimuth(beam_azimuths, beam_energies),
            beam_azimuths=beam_azimuths,
            beam_speech_energies=beam_energies,
        )

    def _record(self, reading: BoardReading) -> None:
        if reading.speech_azimuth is not None:
            self._recorder.record(
                Measurement(
                    name="respeaker/speech_azimuth_deg", value=math.degrees(reading.speech_azimuth)
                )
            )
        for beam, azimuth, energy in zip(
            BEAMS, reading.beam_azimuths, reading.beam_speech_energies, strict=True
        ):
            self._recorder.record(
                Measurement(name=f"respeaker/{beam}/azimuth_deg", value=math.degrees(azimuth))
            )
            self._recorder.record(Measurement(name=f"respeaker/{beam}/speech_energy", value=energy))


def speaker_azimuth(
    beam_azimuths: tuple[float, float, float, float],
    beam_speech_energies: tuple[float, float, float, float],
) -> float | None:
    """The azimuth of the focused beam that hears the most speech, `None` when neither hears any.

    The chip's own direction of the speaker (`AUDIO_MGR_SELECTED_AZIMUTHS`) is meant to be this,
    but it holds its last value for a while after the speech ends, with every energy at 0, and
    once took the silent beam's: it is not used.
    """
    energy, azimuth = max(zip(beam_speech_energies[:2], beam_azimuths[:2], strict=True))
    return azimuth if energy > 0 else None


@asynccontextmanager
async def open_respeaker(config: ReSpeakerConfig, recorder: Recorder) -> AsyncGenerator[ReSpeaker]:
    """The array, ready to be asked.

    Raises:
        ReSpeakerUnavailableError: If it is not plugged in, or this user may not talk to it.
    """
    control = await asyncio.to_thread(XvfControl)
    try:
        logger.info("Asking the reSpeaker XVF3800 where it hears speech from, as {}", config.frame)
        yield ReSpeaker(config, control, recorder)
    finally:
        await asyncio.to_thread(control.close)
