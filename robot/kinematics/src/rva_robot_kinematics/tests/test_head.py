"""Expected angles are those of Pollen's analytical kinematics.

That is `reachy-mini-rust-kinematics` 1.0.3 with the `kinematics_data.json` of `reachy-mini`
1.14.0, given the same poses less 0.177 m in z, its convention.
"""

import math
from pathlib import Path

import pytest

from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import HeadKinematics, UnreachableHeadPoseError
from rva_robot_schemas.head import HeadPose
from rva_robot_schemas.joints import JointCommand

MJCF = Path(__file__).parent / "reachy_mini_head.xml"
KINEMATICS = HeadKinematics(read_head_geometry(MJCF), max_head_yaw=math.radians(65))
NEUTRAL = 0.177
"""The head's height in Pollen's SDK when upright: 27 mm above the rest."""
JOINTS = ("yaw_body", *(f"stewart_{motor}" for motor in range(1, 7)))


def _pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> HeadPose:
    """A pose with its angles in degrees."""
    return HeadPose(
        x=x,
        y=y,
        z=z,
        roll=math.radians(roll),
        pitch=math.radians(pitch),
        yaw=math.radians(yaw),
    )


def _angles(command: JointCommand) -> tuple[float, ...]:
    assert tuple(target.name for target in command.targets) == JOINTS
    return tuple(target.position for target in command.targets)


def test_the_rest_pose_puts_every_joint_at_zero() -> None:
    rest = _pose(0.0, 0.0, 0.149571, roll=0.0, pitch=0.0, yaw=0.0)

    assert _angles(KINEMATICS.solve(rest, body_yaw=0.0)) == pytest.approx((0.0,) * 7, abs=1e-4)


@pytest.mark.parametrize(
    ("target", "body_yaw", "expected"),
    [
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=0.0),
            0.0,
            (0.0, 0.626547, -0.626543, 0.626540, -0.626536, 0.626537, -0.626539),
            id="neutral",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL + 0.01, roll=0.0, pitch=0.0, yaw=0.0),
            0.0,
            (0.0, 0.874636, -0.874631, 0.874628, -0.874623, 0.874624, -0.874627),
            id="lifted by 1 cm",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=15.0, yaw=0.0),
            0.0,
            (0.0, 0.512924, -0.573542, 0.812365, -0.812361, 0.573536, -0.512917),
            id="looking down",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=-15.0, yaw=0.0),
            0.0,
            (0.0, 0.761186, -0.675900, 0.458439, -0.458434, 0.675895, -0.761177),
            id="looking up",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=10.0, pitch=0.0, yaw=0.0),
            0.0,
            (0.0, 0.714806, -0.748616, 0.655967, -0.599695, 0.516360, -0.539169),
            id="tilted",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=30.0),
            0.0,
            (0.0, 0.819145, -0.528417, 0.819138, -0.528411, 0.819136, -0.528414),
            id="turned left",
        ),
        pytest.param(
            _pose(0.005, -0.004, 0.18, roll=5.0, pitch=-10.0, yaw=20.0),
            0.0,
            (0.0, 0.881718, -0.779549, 0.783720, -0.476675, 0.820374, -0.630192),
            id="all at once",
        ),
        pytest.param(
            _pose(0.005, -0.004, 0.18, roll=5.0, pitch=-10.0, yaw=20.0),
            math.radians(-10),
            (-0.174533, 0.938834, -0.755569, 0.849373, -0.476719, 0.906206, -0.605275),
            id="all at once, the body turned right",
        ),
    ],
)
def test_matches_pollens_kinematics(
    target: HeadPose, body_yaw: float, expected: tuple[float, ...]
) -> None:
    assert _angles(KINEMATICS.solve(target, body_yaw)) == pytest.approx(expected, abs=1e-4)


def test_mirrored_motors_turn_opposite_ways_for_a_mirrored_pose() -> None:
    right = _angles(KINEMATICS.solve(_pose(0.0, 0.0, NEUTRAL, 10.0, 5.0, 0.0), body_yaw=0.0))
    left = _angles(KINEMATICS.solve(_pose(0.0, 0.0, NEUTRAL, -10.0, 5.0, 0.0), body_yaw=0.0))

    # Mirrored across the xz plane, motor 1 is motor 6, 2 is 5 and 3 is 4.
    assert right[1:] == pytest.approx([-angle for angle in reversed(left[1:])], abs=1e-4)


def test_the_body_stays_while_the_head_turns_within_reach() -> None:
    target = _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=30.0)
    turned = _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=-20.0)

    angles = _angles(KINEMATICS.solve(target, body_yaw=math.radians(50)))

    assert angles[0] == pytest.approx(math.radians(50))
    assert angles[1:] == pytest.approx(
        _angles(KINEMATICS.solve(turned, body_yaw=0.0))[1:], abs=1e-4
    )


def test_the_body_follows_a_head_turning_beyond_its_reach() -> None:
    target = _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=100.0)

    angles = _angles(KINEMATICS.solve(target, body_yaw=0.0))

    # The head turns 65° from the body, which turns the remaining 35°.
    assert angles == pytest.approx(
        (math.radians(35), 1.092095, -0.591575, 1.092088, -0.591568, 1.092086, -0.591570),
        abs=1e-4,
    )


def test_the_body_reaches_the_head_the_short_way_round() -> None:
    behind = _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=-170.0)

    angles = _angles(KINEMATICS.solve(behind, body_yaw=math.radians(150)))

    # -170° is 190°, 40° from the body at 150°: it stays.
    assert angles[0] == pytest.approx(math.radians(150))


@pytest.mark.parametrize(
    ("target", "error"),
    [
        pytest.param(
            _pose(0.0, 0.0, 0.21, roll=0.0, pitch=0.0, yaw=0.0),
            "The arm of stewart_1 cannot reach its rod",
            id="too high",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=60.0, yaw=0.0),
            "The arm of stewart_3 cannot reach its rod",
            id="looking too far down",
        ),
        pytest.param(
            _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=50.0, yaw=0.0),
            r"stewart_3 would turn to 1.488 rad, beyond its range \[-0.838, 1.396\]",
            id="a motor beyond its range",
        ),
    ],
)
def test_a_pose_out_of_reach_is_refused(target: HeadPose, error: str) -> None:
    with pytest.raises(UnreachableHeadPoseError, match=error):
        KINEMATICS.solve(target, body_yaw=0.0)


def test_a_pose_that_would_turn_the_body_beyond_its_range_is_refused() -> None:
    stiff = HeadKinematics(read_head_geometry(MJCF), max_head_yaw=math.radians(5))
    behind = _pose(0.0, 0.0, NEUTRAL, roll=0.0, pitch=0.0, yaw=180.0)

    with pytest.raises(UnreachableHeadPoseError, match="needs the body beyond its range"):
        stiff.solve(behind, body_yaw=0.0)
