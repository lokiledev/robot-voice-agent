"""Pointing the head camera at points in space, from where the head was when it saw them."""

import math
from collections import deque
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_robot_kinematics.head import pose_matrix
from rva_robot_schemas.head import GazeTarget, HeadPose

_ROUNDS: Final = 4
"""Rounds of aiming from where the last one put the camera, which sits 7 cm from the point the
head turns about: for a point 0.5 m away or more, each round is at least seven times closer than
the last, and the fourth within a hundredth of a degree."""


class HeadPoseUnknownError(LookupError):
    pass


class HeadTrail:
    """Where the head was lately, to place a point the camera saw a moment ago."""

    def __init__(self, keep_s: float, max_gap_s: float) -> None:
        """Keep poses for `keep_s`, and find one within `max_gap_s` of a moment.

        Args:
            keep_s: How long a pose is kept, in seconds: longer than the camera's pipeline takes
                to place a point.
            max_gap_s: How far the pose nearest a moment may be from it, in seconds: a few joint
                states apart.
        """
        self._keep_s: Final = keep_s
        self._max_gap_s: Final = max_gap_s
        self._poses: Final = deque[tuple[float, HeadPose]]()

    def add(self, timestamp: float, pose: HeadPose) -> None:
        """Note that the head was at `pose` at `timestamp`; timestamps come in order."""
        self._poses.append((timestamp, pose))
        while self._poses[0][0] < timestamp - self._keep_s:
            self._poses.popleft()

    def at(self, timestamp: float) -> HeadPose:
        """The pose noted nearest to `timestamp`.

        Raises:
            HeadPoseUnknownError: If none was noted within the maximum gap of it.
        """
        nearest = min(self._poses, key=lambda noted: abs(noted[0] - timestamp), default=None)
        if nearest is None or abs(nearest[0] - timestamp) > self._max_gap_s:
            raise HeadPoseUnknownError(
                f"No head pose within {self._max_gap_s} s of {timestamp:.3f}"
                + ("" if nearest is None else f"; the nearest is at {nearest[0]:.3f}")
            )
        return nearest[1]


class Gaze:
    """Places what the head camera saw in the base frame, and aims the camera at points there.

    The head keeps one position while it looks around, level: only its pitch and yaw change.
    """

    def __init__(
        self, camera: npt.NDArray[np.float64], position: tuple[float, float, float]
    ) -> None:
        """Aim from `position`.

        Args:
            camera: 4×4 pose of the camera's optical frame in the head frame, as
                `HeadGeometry.camera` gives it.
            position: Where the head frame stays, in the base frame, in meters.
        """
        self._camera: Final = camera
        self._position: Final = np.array(position)

    def place(self, target: GazeTarget, head: HeadPose) -> npt.NDArray[np.float64]:
        """The target's point in the base frame, seen with the head at `head`, in meters."""
        seen = np.array((target.x, target.y, target.z, 1.0))
        return (pose_matrix(head) @ self._camera @ seen)[:3]

    def aim(self, point: npt.NDArray[np.float64]) -> HeadPose:
        """The head pose that puts `point`, in the base frame, in the middle of the image.

        A point too high or too low for the camera's axis to reach is looked at as near as it
        reaches; the kinematics refuse what the platform cannot do.
        """
        axis, origin = self._camera[:3, 2], self._camera[:3, 3]
        rotation = np.eye(3)
        pitch = yaw = 0.0
        for _ in range(_ROUNDS):
            direction = point - (self._position + rotation @ origin)
            pitch, yaw = _orient(axis, direction / np.linalg.norm(direction))
            rotation = pose_matrix(_level(self._position, pitch, yaw))[:3, :3]
        return _level(self._position, pitch, yaw)


def _orient(axis: npt.NDArray[np.float64], towards: npt.NDArray[np.float64]) -> tuple[float, float]:
    """The pitch and yaw, roll left at zero, that turn `axis` to `towards`, both unit vectors.

    Pitching by p turns a = (ax, ay, az) to (ax cos p + az sin p, ay, az cos p - ax sin p), whose
    height is r cos(p + φ) with r = |(ax, az)| and φ = atan2(ax, az): p = acos(tz / r) - φ, the
    root that keeps the axis ahead. The yaw then turns it within the horizontal plane.
    """
    ax, ay, az = axis
    reach = math.hypot(ax, az)
    pitch = math.acos(min(max(towards[2] / reach, -1.0), 1.0)) - math.atan2(ax, az)
    ahead = ax * math.cos(pitch) + az * math.sin(pitch)
    yaw = math.atan2(towards[1], towards[0]) - math.atan2(ay, ahead)
    return pitch, math.remainder(yaw, math.tau)


def _level(position: npt.NDArray[np.float64], pitch: float, yaw: float) -> HeadPose:
    x, y, z = position
    return HeadPose(x=float(x), y=float(y), z=float(z), roll=0.0, pitch=pitch, yaw=yaw)
