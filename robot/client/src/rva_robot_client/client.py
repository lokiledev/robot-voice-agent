import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Final

import kickmsg

from rva_pydantic import RVABaseModel
from rva_robot_schemas.camera import CameraFrame, decode_frame
from rva_robot_schemas.joints import JointCommand, JointState
from rva_robot_schemas.topics import CAMERA, JOINT_COMMAND, JOINT_STATE, ROBOT_NODE

_POLL_S: Final = 0.1
"""Longest a receive blocks its thread, so that cancelling a reader takes effect soon."""


class RobotClientConfig(RVABaseModel):
    namespace: str
    """kickmsg namespace of the robot's topics, as the robot is configured."""
    connect_timeout_s: float
    """How long to wait for the robot to be up."""
    silence_timeout_s: float
    """How long a topic may go without a message before the robot counts as gone."""


class RobotUnavailableError(RuntimeError):
    pass


class RobotClient:
    """The robot's joints and head camera, over kickmsg, from any process of the machine."""

    def __init__(self, node: kickmsg.Node, commands: kickmsg.Publisher, silence_s: float) -> None:
        self._node: Final = node
        self._commands: Final = commands
        self._silence_s: Final = silence_s

    def command(self, command: JointCommand) -> None:
        """Send `command` without waiting; the robot applies it at its next step.

        Raises:
            BlockingIOError: If the robot has stopped taking commands and its mailbox is full.
        """
        self._commands.send(command.model_dump_json().encode())

    @asynccontextmanager
    async def joint_states(self) -> AsyncGenerator[AsyncIterator[JointState]]:
        """Subscribe to the joint states; each one yielded is the latest, older ones are skipped.

        The iterator raises `RobotUnavailableError` once the robot has been silent too long.
        """
        subscriber = self._node.subscribe(JOINT_STATE)
        yield self._latest(subscriber, JOINT_STATE, _decode_state)

    @asynccontextmanager
    async def frames(self) -> AsyncGenerator[AsyncIterator[CameraFrame]]:
        """Subscribe to the head camera; each frame yielded is the latest, older ones are skipped.

        The iterator raises `RobotUnavailableError` once the robot has been silent too long.
        """
        subscriber = self._node.subscribe(CAMERA)
        yield self._latest(subscriber, CAMERA, decode_frame)

    async def _latest[T](
        self, subscriber: kickmsg.Subscriber, topic: str, decode: Callable[[memoryview], T]
    ) -> AsyncGenerator[T]:
        heard = time.monotonic()
        while True:
            message = await asyncio.to_thread(lambda: _receive_latest(subscriber, decode))
            if message is not None:
                heard = time.monotonic()
                yield message
            elif time.monotonic() - heard > self._silence_s:
                raise RobotUnavailableError(
                    f"Nothing on {topic} from the robot for {self._silence_s} s"
                )


@asynccontextmanager
async def open_robot(config: RobotClientConfig, name: str) -> AsyncGenerator[RobotClient]:
    """Connect to the robot, waiting for it to be up.

    Args:
        config: Where the robot is.
        name: The client's node name, as kickmsg's tools show it.

    Raises:
        RobotUnavailableError: If the robot is not up within `config.connect_timeout_s`.
    """
    node = kickmsg.Node(name, config.namespace)
    deadline = time.monotonic() + config.connect_timeout_s
    # The robot creates its mailbox last: once it is there, so is every topic.
    while True:
        try:
            commands = node.open_mailbox(ROBOT_NODE, JOINT_COMMAND)
            break
        except RuntimeError as error:
            if time.monotonic() > deadline:
                raise RobotUnavailableError(
                    f"No robot in kickmsg namespace {config.namespace!r} after"
                    f" {config.connect_timeout_s} s; start the simulation with `uv run lune-sim`"
                ) from error
            await asyncio.sleep(_POLL_S)
    yield RobotClient(node, commands, config.silence_timeout_s)


def _receive_latest[T](
    subscriber: kickmsg.Subscriber, decode: Callable[[memoryview], T]
) -> T | None:
    """The newest message waiting, or the next within `_POLL_S`; `None` if none comes.

    Blocks: run it in a thread.
    """
    view = subscriber.receive_view(_POLL_S)
    if view is None:
        return None
    while (newer := subscriber.try_receive_view()) is not None:
        view.release()
        view = newer
    with view, memoryview(view) as buffer:
        return decode(buffer)


def _decode_state(buffer: memoryview) -> JointState:
    return JointState.model_validate_json(bytes(buffer))
