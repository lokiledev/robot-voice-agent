import math

import pytest

from rva_robot_kinematics.motion import HeadMotion, MotionLimits
from rva_robot_schemas.head import HeadPose

LIMITS = MotionLimits(
    max_speed_deg_s=120,
    max_acceleration_deg_s2=600,
    max_speed_m_s=0.05,
    max_acceleration_m_s2=0.5,
    settle_s=0.15,
)
AHEAD = HeadPose(x=0.0, y=0.0, z=0.177, roll=0.0, pitch=0.0, yaw=0.0)
DT = 0.02


def _turned(yaw_deg: float) -> HeadPose:
    return AHEAD.model_copy(update={"yaw": math.radians(yaw_deg)})


def _run(
    motion: HeadMotion, goal: HeadPose, seconds: float, max_speed: float | None
) -> list[float]:
    """The yaw at each step, in degrees."""
    return [math.degrees(motion.step(goal, DT, max_speed).yaw) for _ in range(round(seconds / DT))]


def test_a_far_goal_is_reached_no_faster_than_the_limits() -> None:
    yaws = _run(HeadMotion(AHEAD, LIMITS), _turned(90), seconds=2.0, max_speed=None)

    speeds = [(b - a) / DT for a, b in zip([0.0, *yaws], yaws, strict=False)]
    accelerations = [(b - a) / DT for a, b in zip([0.0, *speeds], speeds, strict=False)]
    assert max(speeds) == pytest.approx(120)
    assert max(abs(a) for a in accelerations) <= 600 + 1e-6
    assert yaws[-1] == pytest.approx(90, abs=0.5)


def test_it_starts_smoothly_when_the_goal_jumps() -> None:
    first, second = _run(HeadMotion(AHEAD, LIMITS), _turned(60), seconds=2 * DT, max_speed=None)

    # 600 degrees/s² for 20 ms: 12 degrees/s, so 0.24 degrees in the first step.
    assert first == pytest.approx(0.24)
    assert second == pytest.approx(0.72)


def test_a_slower_limit_drifts_it_back() -> None:
    yaws = _run(HeadMotion(_turned(40), LIMITS), AHEAD, seconds=1.0, max_speed=20)

    speeds = [(a - b) / DT for a, b in zip([40.0, *yaws], yaws, strict=False)]
    assert max(speeds) == pytest.approx(20)
    # A second at 20 degrees/s, less the 33 ms it takes to get up to it.
    assert yaws[-1] == pytest.approx(20, abs=0.5)


def test_it_turns_the_short_way_round() -> None:
    yaws = _run(HeadMotion(_turned(170), LIMITS), _turned(-170), seconds=1.0, max_speed=None)

    # Through 180, not back through 0.
    assert math.remainder(math.radians(yaws[-1]), math.tau) == pytest.approx(
        math.radians(-170), abs=math.radians(0.5)
    )
    assert all(yaw >= 170 for yaw in yaws)


def test_held_it_stops_where_told() -> None:
    motion = HeadMotion(AHEAD, LIMITS)
    _run(motion, _turned(90), seconds=0.5, max_speed=None)

    motion.hold(_turned(10))
    first = math.degrees(motion.step(_turned(90), DT, None).yaw)

    # From a standstill again: no speed carried over.
    assert first == pytest.approx(10.24)


def test_a_step_after_a_stall_moves_no_more_than_a_tenth_of_a_second_s() -> None:
    motion = HeadMotion(AHEAD, LIMITS)

    stalled = motion.step(_turned(90), 0.5, None)

    # 600 degrees/s² for 100 ms: 60 degrees/s, so 6 degrees, not the 60 of half a second.
    assert math.degrees(stalled.yaw) == pytest.approx(6.0)
