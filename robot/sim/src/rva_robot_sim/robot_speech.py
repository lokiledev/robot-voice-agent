"""What the robot says, told to the simulation by its agent, for the scenario to wait on.

A kickmsg topic of the simulation's namespace, carrying the agent's `RobotSentence`, `RobotLine`
and `RobotReplyEnded` records as JSON. The simulation creates it with its other topics and removes it when
it stops; the agent joins it.
"""

from typing import Annotated, Final

import kickmsg
from pydantic import Field, TypeAdapter

from rva_schemas.record import Record, RobotLine, RobotReplyEnded, RobotSentence

ROBOT_SPEECH: Final = "robot_speech"
"""`RobotSpeech`, published by the agent: a sentence as it starts, a line once said, and the end
of a reply."""

type RobotSpeech = Annotated[
    RobotSentence | RobotLine | RobotReplyEnded, Field(discriminator="kind")
]

_ROBOT_SPEECH: Final[TypeAdapter[RobotSpeech]] = TypeAdapter(RobotSpeech)
_MAX_JSON: Final = 16 * 1024
"""Bytes a record may take as JSON: a long reply is a few thousand."""


def topic_config() -> kickmsg.Config:
    """The topic's, the same for whoever creates it."""
    config = kickmsg.Config()
    config.max_payload_size = _MAX_JSON
    config.max_subscribers = 4
    # Every record counts, unlike a state: room for a reply's sentences between two reads.
    config.sub_ring_capacity = 64
    config.pool_size = config.max_subscribers * config.sub_ring_capacity + 1
    return config


def decode(message: bytes) -> RobotSpeech:
    """The record `message` carries.

    Raises:
        ValueError: If it is not one.
    """
    return _ROBOT_SPEECH.validate_json(message)


class RobotSpeechPublisher:
    """Implements `Recorder`: publishes what the robot says, and leaves the rest.

    The records come from the dialogue's thread only, which is the only one to publish.
    """

    def __init__(self, namespace: str) -> None:
        """Join the topic of the simulation of `namespace`, or create it if none runs yet."""
        self._publisher: Final = kickmsg.Node("agent", namespace).advertise_or_join(
            ROBOT_SPEECH, topic_config()
        )

    def record(self, entry: Record) -> None:
        match entry:
            case RobotSentence() | RobotLine() | RobotReplyEnded():
                self._publisher.send(entry.model_dump_json().encode())
            case _:
                pass
