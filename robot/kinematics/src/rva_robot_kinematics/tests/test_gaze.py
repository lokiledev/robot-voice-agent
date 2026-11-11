import math
from pathlib import Path

import numpy as np
import pytest

from rva_robot_kinematics.gaze import Gaze, HeadPoseUnknownError, HeadTrail
from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import pose_matrix
from rva_robot_schemas.head import GazeTarget, HeadPose

MJCF = Path(__file__).parent / "reachy_mini_head.xml"
CAMERA = read_head_geometry(MJCF).camera
"""40 mm ahead of the head frame and 52 mm above it, looking along its x axis."""
NEUTRAL = 0.177
GAZE = Gaze(CAMERA, (0.0, 0.0, NEUTRAL))


def _upright(yaw_deg: float) -> HeadPose:
    return HeadPose(x=0.0, y=0.0, z=NEUTRAL, roll=0.0, pitch=0.0, yaw=math.radians(yaw_deg))


def test_a_point_seen_on_the_axis_is_ahead_of_the_camera() -> None:
    target = GazeTarget(timestamp=0.0, x=0.0, y=0.0, z=1.0)

    point = GAZE.place(target, _upright(90))

    # Turned left a quarter: ahead is +y.
    assert point == pytest.approx((0.0, 1.0395, NEUTRAL + 0.0525), abs=1e-5)


def test_a_point_seen_right_and_below_is_right_and_below_in_the_base() -> None:
    target = GazeTarget(timestamp=0.0, x=0.2, y=0.1, z=1.0)

    point = GAZE.place(target, _upright(0))

    assert point == pytest.approx((1.0395, -0.2, NEUTRAL + 0.0525 - 0.1), abs=1e-5)


def test_a_point_at_the_cameras_height_straight_ahead_needs_no_turn() -> None:
    pose = GAZE.aim(np.array((1.5, 0.0, NEUTRAL + 0.0525)))

    # To the MJCF's six digits.
    assert (pose.pitch, pose.yaw) == pytest.approx((0.0, 0.0), abs=1e-5)


@pytest.mark.parametrize(
    "point",
    [
        pytest.param((1.0, 0.5, 0.4), id="left and up"),
        pytest.param((0.6, -0.6, 0.1), id="right and down"),
        pytest.param((-0.8, 0.3, 0.3), id="behind"),
        pytest.param((0.3, 0.0, 0.6), id="close above"),
    ],
)
def test_the_camera_aimed_at_a_point_sees_it_in_the_middle(
    point: tuple[float, float, float],
) -> None:
    pose = GAZE.aim(np.array(point))

    camera = pose_matrix(pose) @ CAMERA
    seen = np.linalg.inv(camera) @ np.array((*point, 1.0))
    assert pose.roll == 0.0
    assert (pose.x, pose.y, pose.z) == (0.0, 0.0, NEUTRAL)
    # Within a hundredth of a degree of the axis, and ahead.
    assert math.degrees(math.atan2(math.hypot(seen[0], seen[1]), seen[2])) < 0.01


def test_aiming_at_what_was_placed_turns_the_head_to_it() -> None:
    target = GazeTarget(timestamp=0.0, x=-0.3, y=0.0, z=1.2)

    # Seen 0.3 m left of the middle of the image, 1.24 m ahead of the head frame, with the head
    # turned 10 degrees left: 13.6 degrees further left.
    pose = GAZE.aim(GAZE.place(target, _upright(10)))

    assert math.degrees(pose.yaw) == pytest.approx(10 + 13.6, abs=0.05)
    assert math.degrees(pose.pitch) == pytest.approx(0.0, abs=1e-3)


def test_the_trail_gives_the_pose_nearest_a_moment() -> None:
    trail = HeadTrail(keep_s=1.0, max_gap_s=0.05)
    for step, yaw in enumerate((0, 10, 20)):
        trail.add(100.0 + step * 0.01, _upright(yaw))

    assert trail.at(100.012) == _upright(10)
    assert trail.at(100.06) == _upright(20)


def test_the_trail_knows_nothing_far_from_what_it_noted() -> None:
    trail = HeadTrail(keep_s=1.0, max_gap_s=0.05)
    trail.add(100.0, _upright(0))

    with pytest.raises(HeadPoseUnknownError, match="the nearest is at 100.000"):
        trail.at(100.2)
    with pytest.raises(HeadPoseUnknownError):
        HeadTrail(keep_s=1.0, max_gap_s=0.05).at(100.0)


def test_the_trail_forgets_poses_older_than_it_keeps() -> None:
    trail = HeadTrail(keep_s=1.0, max_gap_s=0.05)
    trail.add(100.0, _upright(0))
    trail.add(101.5, _upright(10))

    with pytest.raises(HeadPoseUnknownError):
        trail.at(100.0)
