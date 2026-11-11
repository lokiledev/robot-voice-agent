"""The client against the simulation's end of the topics, over shared memory, without Genesis."""

import asyncio
from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import numpy as np
import pytest

from rva_robot_client.client import RobotClientConfig, RobotUnavailableError, open_robot
from rva_robot_schemas.camera import CameraFrame
from rva_robot_schemas.joints import JointCommand, JointReading, JointState, JointTarget
from rva_robot_sim.bus import RobotBus


@pytest.fixture
def namespace() -> Iterator[str]:
    """A namespace of its own for each test, removed after."""
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


def _config(namespace: str) -> RobotClientConfig:
    return RobotClientConfig(namespace=namespace, connect_timeout_s=0.3, silence_timeout_s=0.3)


def _state(timestamp: float, yaw: float) -> JointState:
    return JointState(
        timestamp=timestamp,
        joints=(JointReading(name="yaw_body", position=yaw, velocity=0.0),),
    )


async def test_the_latest_joint_state_is_read_and_commands_reach_the_robot(namespace: str) -> None:
    bus = RobotBus(namespace, (4, 2))
    try:
        async with (
            open_robot(_config(namespace), "test_client") as robot,
            robot.joint_states() as states,
        ):
            bus.publish_state(_state(1.0, 0.1))
            bus.publish_state(_state(2.0, 0.2))
            robot.command(JointCommand(targets=(JointTarget(name="yaw_body", position=0.5),)))

            latest = await anext(states)
            commands = bus.commands()
    finally:
        bus.close()

    assert latest == _state(2.0, 0.2)
    assert commands == [JointCommand(targets=(JointTarget(name="yaw_body", position=0.5),))]


async def test_frames_arrive_whole(namespace: str) -> None:
    bus = RobotBus(namespace, (4, 2))
    image = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    try:
        async with open_robot(_config(namespace), "test_client") as robot, robot.frames() as frames:
            bus.publish_frame(CameraFrame(timestamp=5.0, image=image))

            frame = await anext(frames)
    finally:
        bus.close()

    assert frame.timestamp == 5.0
    np.testing.assert_array_equal(frame.image, image)


async def test_a_robot_that_is_not_up_is_reported(namespace: str) -> None:
    with pytest.raises(RobotUnavailableError, match="start the simulation with `uv run lune-sim`"):
        async with open_robot(_config(namespace), "test_client"):
            pass


async def test_a_robot_gone_silent_is_reported(namespace: str) -> None:
    bus = RobotBus(namespace, (4, 2))
    try:
        async with (
            open_robot(_config(namespace), "test_client") as robot,
            robot.joint_states() as states,
        ):
            with pytest.raises(RobotUnavailableError, match="Nothing on joint_state"):
                await asyncio.wait_for(anext(states), timeout=5)
    finally:
        bus.close()


async def test_invalid_commands_are_left_out(namespace: str) -> None:
    bus = RobotBus(namespace, (4, 2))
    try:
        mailbox = kickmsg.Node("test_client", namespace).open_mailbox("robot", "joint_command")
        mailbox.send(b'{"targets": "nope"}')
        mailbox.send(b'{"targets":[]}')

        commands = bus.commands()
    finally:
        bus.close()

    assert commands == [JointCommand(targets=())]
