import math

import numpy as np
import pytest

from rva_robot_sim.hearing import heard_azimuth
from rva_robot_sim.skinning import placement


def _camera_looking_along_x(height: float) -> np.ndarray:
    """A camera at `height` looking along the world's +x, +z up: like the robot's, head upright."""
    camera = np.eye(4)
    # Its -z along +x, its +y along +z, so its +x along -y.
    camera[:3, :3] = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    camera[:3, 3] = (0.0, 0.0, height)
    return camera


@pytest.mark.parametrize(
    ("point", "azimuth_deg"),
    [
        ((2.0, 0.0, 1.25), 0.0),  # Straight ahead.
        ((1.0, 1.0, 1.25), 45.0),  # Ahead on its left.
        ((0.8, -0.65, 1.6), -39.09),  # Ben, on its right; a head higher does not change it.
        ((-1.0, 1.0, 1.25), 135.0),  # Behind it, on its left.
    ],
)
def test_the_microphones_hear_a_voice_from_where_it_is(
    point: tuple[float, float, float], azimuth_deg: float
) -> None:
    azimuth = heard_azimuth(_camera_looking_along_x(1.25), np.array(point))

    assert math.degrees(azimuth) == pytest.approx(azimuth_deg, abs=0.01)


def test_the_microphones_turn_with_the_head() -> None:
    turned_left = placement(0.0, 0.0, math.radians(30)) @ _camera_looking_along_x(1.25)

    azimuth = heard_azimuth(turned_left, np.array([2.0, 0.0, 1.25]))

    # What was ahead is now on its right.
    assert math.degrees(azimuth) == pytest.approx(-30.0)
