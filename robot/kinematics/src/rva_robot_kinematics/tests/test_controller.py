import math
from pathlib import Path

import pytest

from rva_robot_kinematics.controller import HeadController
from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import HeadKinematics, UnreachableHeadPoseError
from rva_robot_schemas.head import HeadPose

MJCF = Path(__file__).parent / "reachy_mini_head.xml"
KINEMATICS = HeadKinematics(read_head_geometry(MJCF), max_head_yaw=math.radians(30))


def _yaw(degrees: float) -> HeadPose:
    """The head at its rest height, turned by `degrees`."""
    return HeadPose(x=0.0, y=0.0, z=0.149571, roll=0.0, pitch=0.0, yaw=math.radians(degrees))


def _body_yaw_deg(controller: HeadController, target: HeadPose) -> float:
    body, *_ = controller.aim(target).targets
    assert body.name == "yaw_body"
    return math.degrees(body.position)


def test_the_body_follows_a_head_turning_far_and_stays_while_it_looks_back() -> None:
    controller = HeadController(KINEMATICS, body_yaw=0.0)

    # The head may turn 30 degrees from the body.
    yaws = [_body_yaw_deg(controller, _yaw(degrees)) for degrees in (20, 50, 70, 40, 20)]

    assert yaws == pytest.approx([0.0, 20.0, 40.0, 40.0, 40.0])


def test_an_unreachable_target_leaves_the_body_where_it_was() -> None:
    controller = HeadController(KINEMATICS, body_yaw=math.radians(10))
    too_low = HeadPose(x=0.0, y=0.0, z=0.05, roll=0.0, pitch=0.0, yaw=0.0)

    with pytest.raises(UnreachableHeadPoseError):
        controller.aim(too_low)

    assert _body_yaw_deg(controller, _yaw(0)) == pytest.approx(10.0)
