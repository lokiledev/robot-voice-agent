from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import pytest

from rva_robot_sim.bus import RobotBus
from rva_robot_sim.npc import PersonPose, lateral, open_people_mover


@pytest.fixture
def namespace() -> Iterator[str]:
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


@pytest.mark.parametrize(
    ("elapsed", "y"),
    [
        (0.0, 0.0),
        # 0.8 m each side at 0.4 m/s: 2 s to an end, 8 s there and back.
        (1.0, -0.4),
        (2.0, -0.8),
        (3.0, -0.4),
        (4.0, 0.0),
        (6.0, 0.8),
        (7.0, 0.4),
        (8.0, 0.0),
        (10.0, -0.8),
    ],
)
def test_pacing_sideways_goes_to_the_robots_right_first(elapsed: float, y: float) -> None:
    assert lateral((1.1, 0.0), half_width=0.8, speed=0.4, elapsed=elapsed) == pytest.approx(
        (1.1, y)
    )


def test_poses_reach_the_simulation_in_order(namespace: str) -> None:
    assert open_people_mover(namespace, "test_walk") is None
    first = PersonPose(person=0, position=(1.1, 0.0), turned_deg=0.0)
    second = PersonPose(person=0, position=(1.1, -0.1), turned_deg=10.0)

    bus = RobotBus(namespace, (4, 2))
    try:
        mover = open_people_mover(namespace, "test_walk")
        assert mover is not None
        mover.send(first)
        mover.send(second)
        poses = bus.poses()
    finally:
        bus.close()

    assert poses == [first, second]
    assert open_people_mover(namespace, "test_walk") is None
