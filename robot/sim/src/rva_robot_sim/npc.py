"""Controlling the simulation's people, its NPCs, while it runs.

The message that places one, its sender, and the paths they walk. The simulation creates the
mailbox with its topics, on a node of its own: the real robot has no people to move.
"""

import asyncio
import time
from typing import Final

import kickmsg
from loguru import logger

from rva_pydantic import RVABaseModel
from rva_robot_sim.people import PlacedPerson

_RATE_HZ: Final = 30.0
"""Poses sent per second while walking: as many as the head camera's frames."""

SIM_NODE: Final = "sim"
"""The simulation's node, which owns the mailbox of people's poses."""
PERSON_POSE: Final = "person_pose"
"""`PersonPose`, a mailbox of the simulation's node that any client may send to."""


class SimulationStoppedError(RuntimeError):
    pass


class PersonPose(RVABaseModel):
    """Where one of the simulation's people stands, and which way they face."""

    person: int
    """Which of the simulation config's `people`, by its index there."""
    position: tuple[float, float]
    """On the floor, in meters, in the robot's frame: it faces +x, +y is its left."""
    turned_deg: float
    """How far they are turned from facing the robot, in degrees, counterclockwise seen from
    above; 0 faces it."""


class PeopleMover:
    """Sends people's poses to the simulation, from any process of the machine."""

    def __init__(self, mailbox: kickmsg.Publisher) -> None:
        self._mailbox: Final = mailbox

    def send(self, pose: PersonPose) -> None:
        """Send `pose` without waiting; the simulation moves them at its next step.

        Raises:
            BlockingIOError: If the simulation has stopped taking poses and its mailbox is full.
        """
        self._mailbox.send(pose.model_dump_json().encode())


def open_people_mover(namespace: str, name: str) -> PeopleMover | None:
    """A sender of poses to the simulation of `namespace`, or `None` if it is not running.

    Args:
        namespace: kickmsg namespace the simulation uses.
        name: The sender's node name, as kickmsg's tools show it.
    """
    try:
        mailbox = kickmsg.Node(name, namespace).open_mailbox(SIM_NODE, PERSON_POSE)
    except RuntimeError:
        return None
    return PeopleMover(mailbox)


def lateral(
    start: tuple[float, float], half_width: float, speed: float, elapsed: float
) -> tuple[float, float]:
    """Where someone pacing sideways across the robot's view is after `elapsed` seconds.

    They go from `start` to their left, the robot's right, as far as `half_width`, back through
    `start` to as far on the other side, and again, at a steady `speed`, turning back at once.

    Args:
        start: Where they start, in the robot's frame, in meters.
        half_width: How far they go each side of `start`, in meters.
        speed: In meters per second.
        elapsed: Seconds since they started.
    """
    x, y = start
    # A quarter period from the middle to an end: up, down twice as far, up.
    phase = (elapsed * speed / (4 * half_width)) % 1.0
    if phase < 0.25:
        offset = 4 * phase
    elif phase < 0.75:
        offset = 2 - 4 * phase
    else:
        offset = 4 * phase - 4
    return x, y - half_width * offset


async def pace(
    mover: PeopleMover, index: int, person: PlacedPerson, half_width: float, speed: float
) -> None:
    """Pace `person`, the simulation's person `index`, sideways until cancelled.

    Once cancelled, they are put back where they started.

    Args:
        mover: To the simulation.
        index: Of the person in the simulation config's `people`.
        person: As the simulation config places them: they start there.
        half_width: How far they go each side of where they start, in meters.
        speed: In meters per second.

    Raises:
        SimulationStoppedError: If the simulation stops taking poses.
    """
    logger.info(
        "{} paces {:g} m each side of {} at {:g} m/s",
        person.avatar,
        half_width,
        person.position,
        speed,
    )
    start = time.monotonic()
    try:
        while True:
            position = lateral(person.position, half_width, speed, time.monotonic() - start)
            mover.send(PersonPose(person=index, position=position, turned_deg=person.turned_deg))
            await asyncio.sleep(1 / _RATE_HZ)
    except BlockingIOError as error:
        raise SimulationStoppedError("The simulation stopped taking people's poses") from error
    except asyncio.CancelledError:
        mover.send(PersonPose(person=index, position=person.position, turned_deg=person.turned_deg))
        raise
