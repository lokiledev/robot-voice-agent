from typing import Final

import kickmsg
from loguru import logger

from rva_robot_schemas.camera import CameraFrame, encode_frame, encoded_size
from rva_robot_schemas.joints import JointCommand, JointState
from rva_robot_schemas.sound import SoundDirection
from rva_robot_schemas.topics import CAMERA, JOINT_COMMAND, JOINT_STATE, ROBOT_NODE, SOUND_DIRECTION
from rva_robot_sim.npc import PERSON_POSE, SCENARIO_START, SIM_NODE, PersonPose, ScenarioStart
from rva_robot_sim.robot_speech import ROBOT_SPEECH, RobotSpeech, decode, topic_config

_MAX_JSON: Final = 16 * 1024
"""Bytes a joint state or command may take as JSON; a few hundred for Reachy Mini."""
_MAX_SUBSCRIBERS: Final = 8
"""Readers of each topic at once: the perception, a controller, a viewer..."""


class RobotBus:
    """The robot's end of its kickmsg topics, and the simulation's mailboxes.

    It creates them all, publishes joint states, frames and where speech is heard from, and takes
    the commands, the people's poses and the scenario's start that clients send, and what the
    agent says the robot says.
    Topics left by a robot that did not close are replaced, so clients of that robot must reopen
    them. Not thread-safe: use it from one thread.
    """

    def __init__(self, namespace: str, camera_size: tuple[int, int]) -> None:
        """Create the topics.

        Args:
            namespace: kickmsg namespace the clients use too.
            camera_size: Width and height of the frames to publish.
        """
        self._node: Final = kickmsg.Node(ROBOT_NODE, namespace)
        self._sim: Final = kickmsg.Node(SIM_NODE, namespace)
        self._unlink()
        self._states: Final = self._node.advertise(
            JOINT_STATE, _config(_MAX_JSON, _MAX_SUBSCRIBERS, ring=16)
        )
        self._sound_directions: Final = self._node.advertise(
            SOUND_DIRECTION, _config(_MAX_JSON, _MAX_SUBSCRIBERS, ring=16)
        )
        # Two frames in each ring: a reader behind gets the latest, not a backlog, and the pool,
        # sized for every ring full, stays small (2.8 MB a frame at 1280x720).
        self._frames: Final = self._node.advertise(
            CAMERA, _config(encoded_size(*camera_size), _MAX_SUBSCRIBERS, ring=2)
        )
        # Created here so that it is there whenever the agent starts, and removed with the rest.
        self._speech: Final = self._sim.subscribe_or_create(ROBOT_SPEECH, topic_config())
        # Only the simulation reads its mailboxes. The robot's comes last: once it is there, so is
        # everything else.
        self._poses: Final = self._sim.create_mailbox(
            PERSON_POSE, _config(_MAX_JSON, subscribers=1, ring=64)
        )
        self._starts: Final = self._sim.create_mailbox(
            SCENARIO_START, _config(_MAX_JSON, subscribers=1, ring=4)
        )
        self._commands: Final = self._node.create_mailbox(
            JOINT_COMMAND, _config(_MAX_JSON, subscribers=1, ring=64)
        )

    def publish_state(self, state: JointState) -> None:
        self._states.send(state.model_dump_json().encode())

    def publish_sound_direction(self, direction: SoundDirection) -> None:
        self._sound_directions.send(direction.model_dump_json().encode())

    def publish_frame(self, frame: CameraFrame) -> None:
        slot = self._frames.allocate()
        if slot is None:
            # Every slot is held by readers: this frame is dropped, the next one may go through.
            logger.warning("Dropped a camera frame: no free slot")
            return
        with memoryview(slot) as buffer:
            length = encode_frame(frame, buffer)
        slot.publish(length)

    def commands(self) -> list[JointCommand]:
        """The commands received since the last call, oldest first.

        Invalid ones are logged and left out: a client's mistake must not stop the robot.
        """
        commands = []
        while (message := self._commands.try_receive()) is not None:
            assert isinstance(message, bytes)
            try:
                commands.append(JointCommand.model_validate_json(message))
            except ValueError:
                logger.exception("Ignored an invalid joint command: {!r}", message[:200])
        return commands

    def poses(self) -> list[PersonPose]:
        """The people's poses received since the last call, oldest first.

        Invalid ones are logged and left out, like commands.
        """
        poses = []
        while (message := self._poses.try_receive()) is not None:
            assert isinstance(message, bytes)
            try:
                poses.append(PersonPose.model_validate_json(message))
            except ValueError:
                logger.exception("Ignored an invalid person pose: {!r}", message[:200])
        return poses

    def starts(self) -> list[ScenarioStart]:
        """The scenario's starts received since the last call, oldest first.

        Invalid ones are logged and left out, like commands.
        """
        starts = []
        while (message := self._starts.try_receive()) is not None:
            assert isinstance(message, bytes)
            try:
                starts.append(ScenarioStart.model_validate_json(message))
            except ValueError:
                logger.exception("Ignored an invalid scenario start: {!r}", message[:200])
        return starts

    def robot_speech(self) -> list[RobotSpeech]:
        """What the agent said the robot says since the last call, oldest first.

        Invalid records are logged and left out, like commands.
        """
        records = []
        while (message := self._speech.try_receive()) is not None:
            assert isinstance(message, bytes)
            try:
                records.append(decode(message))
            except ValueError:
                logger.exception(
                    "Ignored an invalid record of the robot's speech: {!r}", message[:200]
                )
        return records

    def close(self) -> None:
        """Remove the topics; clients still reading them get nothing more."""
        self._unlink()

    def _unlink(self) -> None:
        self._node.unlink_topic(JOINT_STATE)
        self._node.unlink_topic(CAMERA)
        self._node.unlink_topic(SOUND_DIRECTION)
        self._sim.unlink_topic(ROBOT_SPEECH)
        self._node.unlink_mailbox(JOINT_COMMAND)
        self._sim.unlink_mailbox(PERSON_POSE)
        self._sim.unlink_mailbox(SCENARIO_START)


def _config(max_payload: int, subscribers: int, ring: int) -> kickmsg.Config:
    config = kickmsg.Config()
    config.max_payload_size = max_payload
    config.max_subscribers = subscribers
    config.sub_ring_capacity = ring
    # Enough for every ring full and one more message being written.
    config.pool_size = subscribers * ring + 1
    return config
