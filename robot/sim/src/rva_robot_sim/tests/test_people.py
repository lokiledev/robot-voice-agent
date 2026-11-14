import math

import numpy as np
import pytest

from rva_robot_sim.people import standing, yaw_deg


@pytest.mark.parametrize(
    ("position", "turned_deg", "yaw"),
    [
        # Avatars face -y at rest: in front of the robot, facing back along -x, is -90 degrees.
        ((1.0, 0.0), 0.0, 270.0),
        ((0.0, 1.0), 0.0, 0.0),
        ((1.0, 0.0), 45.0, 315.0),
    ],
)
def test_people_face_the_robot_unless_turned(
    position: tuple[float, float], turned_deg: float, yaw: float
) -> None:
    assert yaw_deg(position, turned_deg) % 360 == pytest.approx(yaw)


def test_someone_standing_in_front_faces_the_robot() -> None:
    transform = standing((1.0, 0.0), 0.0)

    np.testing.assert_allclose(transform[:3, 3], [1.0, 0.0, 0.0])
    # Their front, -y at rest, points back at the robot.
    np.testing.assert_allclose(transform[:3, :3] @ [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-12)
    assert math.isclose(np.linalg.det(transform[:3, :3]), 1.0)
