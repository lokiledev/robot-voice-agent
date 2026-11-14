"""Reachy Mini's Dynamixel servos, as the simulation drives its joints.

A Dynamixel in position mode turns the error between its goal and its position into a PWM duty
cycle, `Position P Gain / 128` per encoder pulse out of a full scale of 885, clipped there: its
voltage. The motor's torque is then its stall torque times that fraction, less what its
back-EMF takes as it turns, down to none at its no-load speed. So however far the goal, a
servo turns no faster than its no-load speed, with no more than its stall torque, as the real
one does. Its gearbox adds inertia and friction to the joint.
"""

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_sim.skinning import Floats

PULSES_PER_RAD: Final = 4096 / math.tau
"""The encoder's resolution, 4096 pulses a turn, as all three models have."""
FULL_PWM: Final = 885
"""The PWM's full scale, the default PWM Limit, which the duty cycle is a fraction of."""
P_GAIN_UNIT: Final = 128
"""Position P Gain is in 1/128 of PWM per pulse of error (ROBOTIS: KPP = KPP(TBL) / 128)."""


class ServoConfig(RVABaseModel):
    """A model of Dynamixel at 5 V, and the joints it drives with the gain written to it."""

    model: str
    """ROBOTIS' name, e.g. `XL330-M288-T`, for reading."""
    joints: tuple[str, ...]
    """The MJCF's actuated joints it drives."""
    stall_torque_nm: float = Field(gt=0)
    """At the output shaft, from ROBOTIS' e-manual."""
    no_load_rpm: float = Field(gt=0)
    """At the output shaft, from ROBOTIS' e-manual."""
    p_gain: int = Field(gt=0)
    """Position P Gain, as written to the servo: Pollen's `hardware_config.yaml`. Its I and D
    gains are 0."""
    armature: float = Field(ge=0)
    """The motor's rotor inertia seen through the gearbox, in kg m²."""
    frictionloss_nm: float = Field(ge=0)
    """The gearbox's dry friction, in N m."""


class PassiveJointsConfig(RVABaseModel):
    """The joints no servo drives: the ball joints at the ends of the Stewart platform's rods."""

    joints: tuple[str, ...]
    """The MJCF's names for them."""
    frictionloss_nm: float = Field(ge=0)
    """Their dry friction, on each axis, in N m."""
    armature: float = Field(ge=0)
    """Inertia added on each axis, in kg m², for the solver: the rods weigh 5 g."""


def back_emf_damping(servo: ServoConfig) -> float:
    """How much torque the motor loses per rad/s it turns: its stall torque at no-load speed."""
    return servo.stall_torque_nm / (servo.no_load_rpm * math.tau / 60)


class Servos:
    """The servos of the joints the simulation drives, in their order."""

    def __init__(self, servos: Sequence[ServoConfig], joints: Sequence[str]) -> None:
        """Pair each of `joints` with the servo that drives it.

        Raises:
            ValueError: If a joint has no servo, or two.
        """
        by_joint = {joint: servo for servo in servos for joint in servo.joints}
        driven = [joint for servo in servos for joint in servo.joints]
        if len(driven) != len(set(driven)) or sorted(driven) != sorted(joints):
            raise ValueError(f"Servos drive {sorted(driven)}, not each of {sorted(joints)} once")
        chosen = [by_joint[joint] for joint in joints]
        self._stall: Final = np.array([servo.stall_torque_nm for servo in chosen])
        self._gain: Final = np.array(
            [servo.p_gain / P_GAIN_UNIT * PULSES_PER_RAD / FULL_PWM for servo in chosen]
        )
        """Duty cycle per radian of error."""
        self._damping: Final = np.array([back_emf_damping(servo) for servo in chosen])
        self.armature: Final = np.array([servo.armature for servo in chosen])
        self.frictionloss: Final = np.array([servo.frictionloss_nm for servo in chosen])

    def torques(self, goals: Floats, positions: Floats, velocities: Floats) -> Floats:
        """The torque each servo gives its joint, in N m, from its goal and how the joint is."""
        duty = np.clip(self._gain * (goals - positions), -1.0, 1.0)
        # Braking against its own motion, the motor would draw more than its stall current: its
        # current limit keeps it there.
        return np.clip(self._stall * duty - self._damping * velocities, -self._stall, self._stall)
