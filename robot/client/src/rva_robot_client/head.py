"""The head controller's mailboxes and topic, over kickmsg: both ends.

Its mailboxes take head poses and gaze targets; its topic publishes where the head camera is.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Final

import kickmsg

from rva_pydantic import RVABaseModel
from rva_robot_client.client import receive_latest
from rva_robot_schemas.head import CameraPoseStamped, GazeTarget, HeadPose
from rva_robot_schemas.topics import CAMERA_POSE, GAZE_TARGET, HEAD_CONTROLLER_NODE, HEAD_TARGET

_MAX_JSON: Final = 1024
"""Bytes a head pose or a gaze target may take as JSON; about a hundred."""
_RING: Final = 16
_POSE_RING: Final = 64
"""Camera poses kept for each reader: over half a second at 100 Hz, as a reader takes them once a
frame."""
_POSE_READERS: Final = 4


class HeadSender[T: RVABaseModel]:
    """Sends head poses or gaze targets to the head controller, from any process of the machine."""

    def __init__(self, mailbox: kickmsg.Publisher) -> None:
        self._mailbox: Final = mailbox

    def send(self, message: T) -> None:
        """Send `message` without waiting; the controller takes the latest one it has.

        Raises:
            BlockingIOError: If the controller has stopped taking them and its mailbox is full.
        """
        self._mailbox.send(message.model_dump_json().encode())


def open_head_sender(namespace: str, name: str) -> HeadSender[HeadPose] | None:
    """A sender of head poses to the head controller of `namespace`, or `None` if it is not running.

    Args:
        namespace: kickmsg namespace the controller uses.
        name: The sender's node name, as kickmsg's tools show it.
    """
    return _open(namespace, name, HEAD_TARGET)


def open_gaze_sender(namespace: str, name: str) -> HeadSender[GazeTarget] | None:
    """A sender of gaze targets to the head controller of `namespace`, or `None` if it is down.

    Args:
        namespace: kickmsg namespace the controller uses.
        name: The sender's node name, as kickmsg's tools show it.
    """
    return _open(namespace, name, GAZE_TARGET)


@asynccontextmanager
async def head_targets(namespace: str) -> AsyncGenerator[AsyncIterator[HeadPose]]:
    """The head controller's mailbox of head poses, created on entry and removed on exit.

    Each pose the iterator yields is the latest received; those sent meanwhile are skipped, since
    only where the head should be now matters. A mailbox left by a controller that did not close
    is replaced.
    """
    async with _mailbox(namespace, HEAD_TARGET, HeadPose) as poses:
        yield poses


@asynccontextmanager
async def gaze_targets(namespace: str) -> AsyncGenerator[AsyncIterator[GazeTarget]]:
    """The head controller's mailbox of gaze targets, created on entry and removed on exit.

    Each target the iterator yields is the latest received, like `head_targets`'s poses.
    """
    async with _mailbox(namespace, GAZE_TARGET, GazeTarget) as targets:
        yield targets


def _open[T: RVABaseModel](namespace: str, name: str, mailbox: str) -> HeadSender[T] | None:
    try:
        publisher = kickmsg.Node(name, namespace).open_mailbox(HEAD_CONTROLLER_NODE, mailbox)
    except RuntimeError:
        return None
    return HeadSender(publisher)


@asynccontextmanager
async def _mailbox[T: RVABaseModel](
    namespace: str, name: str, model: type[T]
) -> AsyncGenerator[AsyncIterator[T]]:
    node = kickmsg.Node(HEAD_CONTROLLER_NODE, namespace)
    node.unlink_mailbox(name)
    config = kickmsg.Config()
    config.max_payload_size = _MAX_JSON
    config.max_subscribers = 1
    config.sub_ring_capacity = _RING
    config.pool_size = _RING + 1
    mailbox = node.create_mailbox(name, config)
    try:
        yield _latest(mailbox, model)
    finally:
        node.unlink_mailbox(name)


async def _latest[T: RVABaseModel](
    mailbox: kickmsg.Subscriber, model: type[T]
) -> AsyncGenerator[T]:
    def decode(buffer: memoryview) -> T:
        return model.model_validate_json(bytes(buffer))

    while True:
        message = await asyncio.to_thread(lambda: receive_latest(mailbox, decode))
        if message is not None:
            yield message


class CameraPosePublisher:
    """Publishes where the head camera is, from the head controller."""

    def __init__(self, topic: kickmsg.Publisher) -> None:
        self._topic: Final = topic

    def publish(self, pose: CameraPoseStamped) -> None:
        self._topic.send(pose.model_dump_json().encode())


@contextmanager
def advertise_camera_poses(namespace: str) -> Generator[CameraPosePublisher]:
    """The head controller's topic of camera poses, created on entry and removed on exit.

    A topic left by a controller that did not close is replaced.
    """
    node = kickmsg.Node(HEAD_CONTROLLER_NODE, namespace)
    node.unlink_topic(CAMERA_POSE)
    config = kickmsg.Config()
    config.max_payload_size = _MAX_JSON
    config.max_subscribers = _POSE_READERS
    config.sub_ring_capacity = _POSE_RING
    config.pool_size = _POSE_READERS * _POSE_RING + 1
    topic = node.advertise(CAMERA_POSE, config)
    try:
        yield CameraPosePublisher(topic)
    finally:
        node.unlink_topic(CAMERA_POSE)


class CameraPoseReader:
    """Takes the camera poses the head controller publishes, without waiting.

    Not thread-safe: use it from one thread.
    """

    def __init__(self, namespace: str, name: str) -> None:
        """Read the controller of `namespace` as node `name`, once it is up."""
        self._node: Final = kickmsg.Node(name, namespace)
        self._subscriber: kickmsg.Subscriber | None = None

    def received(self) -> list[CameraPoseStamped]:
        """The poses published since the last call, oldest first.

        None until the controller is up, nor after it stopped.
        """
        if self._subscriber is None:
            try:
                self._subscriber = self._node.subscribe(CAMERA_POSE)
            except RuntimeError:
                return []
        poses = []
        while (message := self._subscriber.try_receive()) is not None:
            assert isinstance(message, bytes)
            poses.append(CameraPoseStamped.model_validate_json(message))
        return poses
