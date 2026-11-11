from typing import Final

import kickmsg
from loguru import logger

from rva_robot_schemas.camera import CameraFrame, encode_frame, encoded_size
from rva_robot_schemas.joints import JointCommand, JointState
from rva_robot_schemas.topics import CAMERA, JOINT_COMMAND, JOINT_STATE, ROBOT_NODE

_MAX_JSON: Final = 16 * 1024
"""Bytes a joint state or command may take as JSON; a few hundred for Reachy Mini."""
_MAX_SUBSCRIBERS: Final = 8
"""Readers of each topic at once: the perception, a controller, a viewer..."""


class RobotBus:
    """The robot's end of its kickmsg topics.

    It creates them all, publishes joint states and frames, and takes the commands clients send.
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
        self._unlink()
        self._states: Final = self._node.advertise(
            JOINT_STATE, _config(_MAX_JSON, _MAX_SUBSCRIBERS, ring=16)
        )
        # Two frames in each ring: a reader behind gets the latest, not a backlog, and the pool,
        # sized for every ring full, stays small (2.8 MB a frame at 1280x720).
        self._frames: Final = self._node.advertise(
            CAMERA, _config(encoded_size(*camera_size), _MAX_SUBSCRIBERS, ring=2)
        )
        # Only the robot reads its mailbox.
        self._commands: Final = self._node.create_mailbox(
            JOINT_COMMAND, _config(_MAX_JSON, subscribers=1, ring=64)
        )

    def publish_state(self, state: JointState) -> None:
        self._states.send(state.model_dump_json().encode())

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

    def close(self) -> None:
        """Remove the topics; clients still reading them get nothing more."""
        self._unlink()

    def _unlink(self) -> None:
        self._node.unlink_topic(JOINT_STATE)
        self._node.unlink_topic(CAMERA)
        self._node.unlink_mailbox(JOINT_COMMAND)


def _config(max_payload: int, subscribers: int, ring: int) -> kickmsg.Config:
    config = kickmsg.Config()
    config.max_payload_size = max_payload
    config.max_subscribers = subscribers
    config.sub_ring_capacity = ring
    # Enough for every ring full and one more message being written.
    config.pool_size = subscribers * ring + 1
    return config
