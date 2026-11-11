"""The head controller's mailbox of head poses, over kickmsg: its end and the senders'."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

import kickmsg

from rva_robot_client.client import receive_latest
from rva_robot_schemas.head import HeadPose
from rva_robot_schemas.topics import HEAD_CONTROLLER_NODE, HEAD_TARGET

_MAX_JSON: Final = 1024
"""Bytes a head pose may take as JSON; about a hundred."""
_RING: Final = 16


class HeadSender:
    """Sends head poses to the head controller, from any process of the machine."""

    def __init__(self, mailbox: kickmsg.Publisher) -> None:
        self._mailbox: Final = mailbox

    def send(self, pose: HeadPose) -> None:
        """Send `pose` without waiting; the controller takes the latest one it has.

        Raises:
            BlockingIOError: If the controller has stopped taking poses and its mailbox is full.
        """
        self._mailbox.send(pose.model_dump_json().encode())


def open_head_sender(namespace: str, name: str) -> HeadSender | None:
    """A sender to the head controller of `namespace`, or `None` if it is not running.

    Args:
        namespace: kickmsg namespace the controller uses.
        name: The sender's node name, as kickmsg's tools show it.
    """
    try:
        mailbox = kickmsg.Node(name, namespace).open_mailbox(HEAD_CONTROLLER_NODE, HEAD_TARGET)
    except RuntimeError:
        return None
    return HeadSender(mailbox)


@asynccontextmanager
async def head_targets(namespace: str) -> AsyncGenerator[AsyncIterator[HeadPose]]:
    """The head controller's mailbox, created on entry and removed on exit.

    Each pose the iterator yields is the latest received; those sent meanwhile are skipped, since
    only where the head should be now matters. A mailbox left by a controller that did not close
    is replaced.
    """
    node = kickmsg.Node(HEAD_CONTROLLER_NODE, namespace)
    node.unlink_mailbox(HEAD_TARGET)
    config = kickmsg.Config()
    config.max_payload_size = _MAX_JSON
    config.max_subscribers = 1
    config.sub_ring_capacity = _RING
    config.pool_size = _RING + 1
    mailbox = node.create_mailbox(HEAD_TARGET, config)
    try:
        yield _poses(mailbox)
    finally:
        node.unlink_mailbox(HEAD_TARGET)


async def _poses(mailbox: kickmsg.Subscriber) -> AsyncGenerator[HeadPose]:
    while True:
        pose = await asyncio.to_thread(lambda: receive_latest(mailbox, _decode))
        if pose is not None:
            yield pose


def _decode(buffer: memoryview) -> HeadPose:
    return HeadPose.model_validate_json(bytes(buffer))
