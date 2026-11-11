"""The joint angles that put Reachy Mini's head in a pose, in closed form."""

import math

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel
from rva_robot_kinematics.geometry import HeadGeometry, StewartLeg
from rva_robot_schemas.joints import JointCommand, JointTarget


class HeadPose(RVABaseModel):
    """The pose of the head frame, the MJCF's `head` site, in the robot's base frame.

    The base frame is the MJCF's root: on the table top, under the body's axis, with x forward, y
    to the left and z up. The head frame has x forward and z up too; the camera sits 4 cm ahead of
    it and 5 cm above. With every joint at zero, the head frame is the base frame raised by
    0.1496 m. Pollen's SDK holds the head 0.177 m high by default, and gives its poses less that
    height.

    The rotation turns by `yaw` about z, then by `pitch` about the turned y, then by `roll` about
    the turned x: R = Rz(yaw) Ry(pitch) Rx(roll), as Pollen's SDK builds head poses.
    """

    x: float
    """In meters."""
    y: float
    """In meters."""
    z: float
    """In meters."""
    roll: float
    """In radians; positive tilts the head to its right."""
    pitch: float
    """In radians; positive looks down."""
    yaw: float
    """In radians; positive turns to the left."""


class UnreachableHeadPoseError(ValueError):
    pass


class HeadKinematics:
    """Puts the head in a pose with the body's yaw and the six motors of the Stewart platform.

    The platform can turn the head only so far from the body; the body's yaw turns it further.
    """

    def __init__(self, geometry: HeadGeometry, max_head_yaw: float) -> None:
        """Kinematics with a limit on the head's yaw from the body.

        Args:
            geometry: Of the robot, as its MJCF describes it.
            max_head_yaw: How far the platform may turn the head from the body, either way, in
                radians; Pollen's SDK allows 65°.
        """
        self._geometry = geometry
        self._max_head_yaw = max_head_yaw
        self._body_inverse = np.linalg.inv(geometry.body)
        self._motor_inverses = tuple(
            np.linalg.inv(leg.motor).astype(np.float64) for leg in geometry.legs
        )

    def solve(self, target: HeadPose, body_yaw: float) -> JointCommand:
        """The angles of the body's joint and of the motors that put the head at `target`.

        Args:
            target: Where the head should be.
            body_yaw: The body's yaw to keep, in radians, e.g. its current one: it turns only as
                much as keeps the head within the maximum yaw from it. The body then follows a
                head that turns far, and stays still while the head looks around in front of it.

        Raises:
            UnreachableHeadPoseError: If the body would turn beyond its range, a motor's arm cannot
                reach its rod at any angle, or a motor would turn beyond its range.
        """
        yaw = self._body_yaw(target.yaw, body_yaw)
        head = _rotation_z(-yaw) @ self._body_inverse @ _transform(target)
        angles = (
            self._arm_angle(leg, motor_inverse, head)
            for leg, motor_inverse in zip(self._geometry.legs, self._motor_inverses, strict=True)
        )
        return JointCommand(
            targets=(
                JointTarget(name=self._geometry.body_joint, position=yaw),
                *(
                    JointTarget(name=leg.joint, position=angle)
                    for leg, angle in zip(self._geometry.legs, angles, strict=True)
                ),
            )
        )

    def _body_yaw(self, head_yaw: float, keep: float) -> float:
        """The body's yaw nearest to `keep` within the maximum yaw from the head's, in its range."""
        lower, upper = self._geometry.body_range
        # A heading a whole turn away is the same: the body may reach the head either way round.
        headings = (math.remainder(head_yaw, math.tau) + turns * math.tau for turns in (-1, 0, 1))
        yaws = (
            min(max(keep, heading - self._max_head_yaw), heading + self._max_head_yaw)
            for heading in headings
        )
        reachable = [yaw for yaw in yaws if lower <= yaw <= upper]
        if not reachable:
            raise UnreachableHeadPoseError(
                f"The head at yaw {head_yaw:.3f} rad needs the body beyond its range"
                f" [{lower:.3f}, {upper:.3f}]"
            )
        return min(reachable, key=lambda yaw: abs(yaw - keep))

    def _arm_angle(
        self,
        leg: StewartLeg,
        motor_inverse: npt.NDArray[np.float64],
        head: npt.NDArray[np.float64],
    ) -> float:
        """The angle of the motor that puts its arm's tip at rod length from the rod's anchor.

        The tip turns on a circle about the motor's z axis: at angle q, it is at
        r (cos q, sin q, 0). It is at rod length L from the anchor p when
        px cos q + py sin q = (|p|² + r² - L²) / 2r, that is when
        cos(q - φ) = (|p|² + r² - L²) / 2r / ρ, with p = ρ (cos φ, sin φ, pz).
        """
        x, y, z = (motor_inverse @ head @ np.append(leg.anchor, 1.0))[:3]
        arm, rod = self._geometry.arm_length, self._geometry.rod_length
        reach = (x * x + y * y + z * z + arm * arm - rod * rod) / (2 * arm)
        radius = math.hypot(x, y)
        if abs(reach) > radius:
            raise UnreachableHeadPoseError(f"The arm of {leg.joint} cannot reach its rod")
        angle = math.remainder(math.atan2(y, x) + leg.branch * math.acos(reach / radius), math.tau)
        lower, upper = leg.range
        if not lower <= angle <= upper:
            raise UnreachableHeadPoseError(
                f"{leg.joint} would turn to {angle:.3f} rad, beyond its range"
                f" [{lower:.3f}, {upper:.3f}]"
            )
        return angle


def _transform(pose: HeadPose) -> npt.NDArray[np.float64]:
    cos_pitch, sin_pitch = math.cos(pose.pitch), math.sin(pose.pitch)
    cos_roll, sin_roll = math.cos(pose.roll), math.sin(pose.roll)
    pitch = np.array(((cos_pitch, 0, sin_pitch), (0, 1, 0), (-sin_pitch, 0, cos_pitch)))
    roll = np.array(((1, 0, 0), (0, cos_roll, -sin_roll), (0, sin_roll, cos_roll)))
    transform = _rotation_z(pose.yaw)
    transform[:3, :3] = transform[:3, :3] @ pitch @ roll
    transform[:3, 3] = (pose.x, pose.y, pose.z)
    return transform


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    transform = np.eye(4)
    cos, sin = math.cos(angle), math.sin(angle)
    transform[:2, :2] = ((cos, -sin), (sin, cos))
    return transform
