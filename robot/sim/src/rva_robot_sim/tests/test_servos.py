import math

import numpy as np
import pytest

from rva_robot_sim.servos import ServoConfig, Servos, back_emf_damping

XL330_M288 = ServoConfig(
    model="XL330-M288-T",
    joints=("stewart_1",),
    stall_torque_nm=0.52,
    no_load_rpm=103,
    p_gain=300,
    armature=0.0018,
    frictionloss_nm=0.058,
)
NO_LOAD = 103 * math.tau / 60
"""The XL330-M288's no-load speed, in rad/s."""


def _torque(goal: float, position: float, velocity: float) -> float:
    servos = Servos([XL330_M288], ["stewart_1"])
    return float(servos.torques(np.array([goal]), np.array([position]), np.array([velocity]))[0])


def test_far_from_its_goal_it_gives_its_stall_torque() -> None:
    assert _torque(goal=1.0, position=0.0, velocity=0.0) == pytest.approx(0.52)
    assert _torque(goal=-1.0, position=0.0, velocity=0.0) == pytest.approx(-0.52)


def test_it_turns_no_faster_than_its_no_load_speed() -> None:
    assert _torque(goal=1.0, position=0.0, velocity=NO_LOAD) == pytest.approx(0.0)
    # Faster, it brakes.
    assert _torque(goal=1.0, position=0.0, velocity=1.5 * NO_LOAD) < 0


def test_near_its_goal_its_torque_grows_with_the_error() -> None:
    # 300 / 128 of PWM per pulse, 4096 / 2π pulses a radian, over 885 of full PWM.
    stiffness = 0.52 * 300 / 128 * 4096 / math.tau / 885

    assert _torque(goal=0.01, position=0.0, velocity=0.0) == pytest.approx(0.01 * stiffness)
    assert stiffness == pytest.approx(0.898, abs=0.001)


def test_braking_it_draws_no_more_than_its_stall_current() -> None:
    assert _torque(goal=-1.0, position=0.0, velocity=NO_LOAD) == pytest.approx(-0.52)


def test_back_emf_takes_its_stall_torque_at_its_no_load_speed() -> None:
    assert back_emf_damping(XL330_M288) * NO_LOAD == pytest.approx(0.52)


def test_each_joint_has_one_servo() -> None:
    with pytest.raises(ValueError, match="not each of"):
        Servos([XL330_M288], ["stewart_1", "stewart_2"])
    with pytest.raises(ValueError, match="not each of"):
        Servos([XL330_M288, XL330_M288], ["stewart_1"])
