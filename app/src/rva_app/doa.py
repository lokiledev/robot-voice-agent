"""Direction of arrival: where the robot's microphones hear speech from, whichever they are."""

import math
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Final, Literal, assert_never

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_client.client import RobotClient, RobotClientConfig, open_robot
from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.sound import SoundDirections
from rva_schemas.record import Measurement
from rva_schemas.sound import HeardDirection
from rva_voice_respeaker.config import ReSpeakerConfig
from rva_voice_respeaker.respeaker import open_respeaker


class RobotDoaConfig(RVABaseModel):
    """The robot's own microphones, through the robot client.

    For now the simulation's, whose array is in the head at the camera (`rva_robot_sim.hearing`).
    """

    kind: Literal["robot"] = "robot"
    frame: str
    """The array's link, which the robot reports its azimuths in."""


type DoaConfig = Annotated[ReSpeakerConfig | RobotDoaConfig, Field(discriminator="kind")]


class RobotSoundDirections:
    """`SoundDirections` from the robot's microphones, over its kickmsg topic."""

    def __init__(self, robot: RobotClient, frame: str, recorder: Recorder) -> None:
        self._robot: Final = robot
        self._frame: Final = frame
        self._recorder: Final = recorder

    async def directions(self) -> AsyncGenerator[HeardDirection]:
        """Each direction the robot reports, recorded too, until cancelled.

        Raises:
            RobotUnavailableError: Once the robot has been silent too long.
        """
        async with self._robot.sound_directions() as directions:
            async for direction in directions:
                if direction.speech_azimuth is not None:
                    self._recorder.record(
                        Measurement(
                            name="robot/speech_azimuth_deg",
                            value=math.degrees(direction.speech_azimuth),
                        )
                    )
                yield HeardDirection(
                    time=direction.timestamp,
                    frame=self._frame,
                    speech_azimuth=direction.speech_azimuth,
                )


def open_sound_directions(
    config: DoaConfig, robot: RobotClientConfig | None, recorder: Recorder
) -> AbstractAsyncContextManager[SoundDirections]:
    """The microphone array `config` describes, ready to tell where it hears speech from.

    Raises:
        ReSpeakerUnavailableError: If it is the reSpeaker, and it is not plugged in.
        RobotUnavailableError: If it is the robot's, and the robot is not up.
    """
    match config:
        case ReSpeakerConfig():
            return open_respeaker(config, recorder)
        case RobotDoaConfig():
            assert robot is not None, "validated with the runtime config"
            return _robot_sound_directions(robot, config, recorder)
        case _:
            assert_never(config)


@asynccontextmanager
async def _robot_sound_directions(
    robot: RobotClientConfig, config: RobotDoaConfig, recorder: Recorder
) -> AsyncGenerator[RobotSoundDirections]:
    async with open_robot(robot, "agent_doa") as client:
        yield RobotSoundDirections(client, config.frame, recorder)
