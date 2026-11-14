"""How the head moves between the poses it is sent: smoothly, no faster than a limit."""

import math
from typing import Final

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_schemas.head import HeadPose

_ANGLES: Final = ("roll", "pitch", "yaw")
_POSITIONS: Final = ("x", "y", "z")


class MotionLimits(RVABaseModel):
    """How fast and how abruptly the head may move, on each axis."""

    max_speed_deg_s: float = Field(gt=0)
    """Of each angle."""
    max_acceleration_deg_s2: float = Field(gt=0)
    """Of each angle: how fast it starts and stops."""
    max_speed_m_s: float = Field(gt=0)
    """Of each coordinate."""
    max_acceleration_m_s2: float = Field(gt=0)
    settle_s: float = Field(gt=0)
    """How long it takes to close most of the way to the goal once near: the time constant of
    the approach, which slows the head down as it gets there."""


class HeadMotion:
    """Moves the pose sent to the head towards a goal, within the limits, step by step.

    Each axis heads for the goal at a speed that falls as it nears it, within the speed limit,
    and changes that speed no faster than the acceleration limit: the head starts, turns and
    stops smoothly, whatever the goals, however they jump.
    """

    def __init__(self, start: HeadPose, limits: MotionLimits) -> None:
        """Start still, at `start`."""
        self._limits: Final = limits
        self._pose = start
        self._velocity = dict.fromkeys(_ANGLES + _POSITIONS, 0.0)

    @property
    def pose(self) -> HeadPose:
        return self._pose

    def step(self, goal: HeadPose, dt: float, max_speed_deg_s: float | None) -> HeadPose:
        """The pose `dt` seconds on, heading for `goal`.

        Args:
            goal: Where the head should end up.
            dt: Seconds since the last step.
            max_speed_deg_s: A lower speed limit for the angles than the limits', e.g. to drift
                back slowly; `None` for the limits'.
        """
        limits = self._limits
        angular_speed = math.radians(
            limits.max_speed_deg_s
            if max_speed_deg_s is None
            else min(max_speed_deg_s, limits.max_speed_deg_s)
        )
        angular_acceleration = math.radians(limits.max_acceleration_deg_s2)
        moved = {}
        for axis in _ANGLES + _POSITIONS:
            angular = axis in _ANGLES
            error = getattr(goal, axis) - getattr(self._pose, axis)
            if axis == "yaw":
                # The short way round.
                error = math.remainder(error, math.tau)
            speed = angular_speed if angular else limits.max_speed_m_s
            acceleration = angular_acceleration if angular else limits.max_acceleration_m_s2
            wanted = _clamp(error / limits.settle_s, speed)
            velocity = self._velocity[axis]
            velocity += _clamp(wanted - velocity, acceleration * dt)
            self._velocity[axis] = velocity
            moved[axis] = getattr(self._pose, axis) + velocity * dt
        self._pose = HeadPose(**moved)
        return self._pose

    def hold(self, pose: HeadPose) -> None:
        """Stop at `pose`: where the head last could go, when the next step's cannot."""
        self._pose = pose
        self._velocity = dict.fromkeys(self._velocity, 0.0)


def _clamp(value: float, limit: float) -> float:
    return min(max(value, -limit), limit)
