from pathlib import Path

import numpy as np
import pytest

from rva_robot_kinematics.geometry import read_head_geometry

# The kinematic tree of Reachy Mini's MJCF, as the simulation downloads it, without its geometry.
MJCF = Path(__file__).parent / "reachy_mini_head.xml"


def test_reads_the_stewart_platform_of_reachy_mini() -> None:
    geometry = read_head_geometry(MJCF)

    assert geometry.body_joint == "yaw_body"
    assert geometry.body_range == pytest.approx((-2.792527, 2.792527))
    assert geometry.arm_length == pytest.approx(0.04)
    assert geometry.rod_length == pytest.approx(0.085)
    legs = geometry.legs
    assert [leg.joint for leg in legs] == [f"stewart_{motor}" for motor in range(1, 7)]
    # Mounted in mirrored pairs, 1 and 6, 2 and 5, 3 and 4, each motor bends its arm the other
    # way to its neighbor's.
    assert [leg.branch for leg in legs] == [-1, 1, -1, 1, -1, 1]
    assert np.array([leg.range for leg in legs]) == pytest.approx(
        np.array(
            [
                (-0.837758, 1.396263),
                (-1.396263, 1.221730),
                (-0.837758, 1.396263),
                (-1.396263, 0.837758),
                (-1.221730, 1.396263),
                (-1.396263, 0.837758),
            ]
        )
    )
    # The anchors are in the head's plane, under the camera; the motors, about 77 mm high, all in
    # the base's frame.
    assert np.array([leg.anchor for leg in legs]) == pytest.approx(
        np.array(
            [
                (0.020648, 0.021764, 0.0),
                (0.008524, 0.028764, 0.0),
                (-0.029172, 0.007, 0.0),
                (-0.029172, -0.007, 0.0),
                (0.008524, -0.028764, 0.0),
                (0.020648, -0.021764, 0.0),
            ]
        ),
        abs=1e-6,
    )
    assert np.array([(geometry.body @ leg.motor)[:3, 3] for leg in legs]) == pytest.approx(
        np.array(
            [
                (0.026991, 0.026749, 0.076633),
                (0.009670, 0.036749, 0.076633),
                (-0.036661, 0.01, 0.076633),
                (-0.036661, -0.01, 0.076633),
                (0.009670, -0.036749, 0.076633),
                (0.026991, -0.026749, 0.076633),
            ]
        ),
        abs=1e-6,
    )


def test_a_frame_oriented_otherwise_than_by_a_quaternion_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reachy_mini.xml"
    path.write_text(
        MJCF.read_text().replace(
            '<body name="body_foot_3dprint" pos="0 0 0" quat="1 0 0 0">',
            '<body name="body_foot_3dprint" pos="0 0 0" euler="0 0 0">',
        )
    )

    with pytest.raises(ValueError, match="body_foot_3dprint is oriented by euler, not by quat"):
        read_head_geometry(path)


def test_a_rod_that_misses_the_head_at_rest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reachy_mini.xml"
    path.write_text(
        MJCF.read_text().replace(
            '<site name="closing_3_1" pos="-0.085 ', '<site name="closing_3_1" pos="-0.09 '
        )
    )

    with pytest.raises(ValueError, match="With every joint at zero, the rod of stewart_3 misses"):
        read_head_geometry(path)
