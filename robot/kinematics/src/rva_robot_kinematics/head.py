"""Reachy Mini's head kinematics: the joint angles for a pose in closed form, and back by Newton."""

import math
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_robot_kinematics.geometry import HeadGeometry, StewartLeg
from rva_robot_schemas.head import HeadPose
from rva_robot_schemas.joints import JointCommand, JointTarget

_TOLERANCE: Final = 1e-10
"""How far, in meters, each rod may be from its length for a pose to fit the joints."""
_MAX_ITERATIONS: Final = 20
"""Newton's steps before giving up; from a pose within a few degrees, it takes two or three."""
_MAX_TRANSLATION_STEP: Final = 0.01
"""How far, in meters, one of Newton's steps may move the head."""
_MAX_ROTATION_STEP: Final = 0.3
"""How far, in radians, one of Newton's steps may turn the head."""


class UnreachableHeadPoseError(ValueError):
    pass


class HeadPoseNotFoundError(ValueError):
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
        # Stacked, for the forward kinematics to handle the six legs at once.
        self._arm_axes = (
            np.stack([leg.motor[:3, :2] for leg in geometry.legs]) * geometry.arm_length
        )
        self._motor_origins = np.stack([leg.motor[:3, 3] for leg in geometry.legs])
        self._anchors = np.stack([leg.anchor for leg in geometry.legs])

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

    def forward(self, positions: Mapping[str, float], guess: HeadPose | None) -> HeadPose:
        """The pose the head is in with the joints at `positions`.

        The six rods close loops that have no closed form: Newton's method moves the head from
        `guess` until each rod's anchor is at rod length from its arm's tip. From the last
        estimate, a joint state earlier, it converges in two or three steps.

        The joints' angles fit several poses, all but one of which the robot cannot be assembled
        in. Newton's method finds the one nearest the guess; it moves the head by small steps so as
        not to jump to another.

        Args:
            positions: Of the body's joint and of the six motors, by name, in radians, e.g. from a
                `JointState`; other joints are ignored.
            guess: A pose near the head's, e.g. the last estimate. `None` without one starts from
                the pose with every motor at zero, turned with the body, which leads to the robot's
                pose wherever the motors' ranges allow; a guess off by tens of degrees, such as an
                estimate from before the body turned, may not.

        Returns:
            The pose, with its yaw in [-π, π].

        Raises:
            KeyError: If `positions` lacks one of the joints.
            HeadPoseNotFoundError: If Newton's method does not converge: no pose fits the angles,
                or none near the guess.
        """
        yaw = positions[self._geometry.body_joint]
        angles = np.array([positions[leg.joint] for leg in self._geometry.legs])
        arms = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        tips = np.einsum("lij,lj->li", self._arm_axes, arms) + self._motor_origins
        head = (
            self._geometry.head
            if guess is None
            else _rotation_z(-yaw) @ self._body_inverse @ _transform(guess)
        )
        rotation, translation = head[:3, :3], head[:3, 3]
        rod = self._geometry.rod_length
        for _ in range(_MAX_ITERATIONS):
            anchors = self._anchors @ rotation.T
            rods = anchors + translation - tips
            # Each rod's length less the rod length, to first order, without a square root.
            errors = (np.einsum("li,li->l", rods, rods) - rod * rod) / (2 * rod)
            if np.max(np.abs(errors)) < _TOLERANCE:
                break
            # A small translation dt and rotation dω move an anchor by dt + dω × a.
            jacobian = np.hstack((rods, np.cross(anchors, rods))) / rod
            try:
                step = np.linalg.solve(jacobian, -errors)
            except np.linalg.LinAlgError as error:
                raise HeadPoseNotFoundError("The platform is singular at the guess") from error
            # A full step from afar may land near a pose the robot cannot be assembled in.
            moved, turned = np.linalg.norm(step[:3]), np.linalg.norm(step[3:])
            step *= min(
                _MAX_TRANSLATION_STEP / max(moved, _MAX_TRANSLATION_STEP),
                _MAX_ROTATION_STEP / max(turned, _MAX_ROTATION_STEP),
            )
            translation = translation + step[:3]
            rotation = _rotation(step[3:]) @ rotation
        else:
            raise HeadPoseNotFoundError(
                f"No head pose fits the joints within {_MAX_ITERATIONS} steps from the guess"
            )
        head = np.eye(4)
        head[:3, :3], head[:3, 3] = rotation, translation
        return _pose(self._geometry.body @ _rotation_z(yaw) @ head)

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


def _pose(transform: npt.NDArray[np.float64]) -> HeadPose:
    rotation = transform[:3, :3]
    x, y, z = transform[:3, 3]
    return HeadPose(
        x=float(x),
        y=float(y),
        z=float(z),
        roll=math.atan2(rotation[2, 1], rotation[2, 2]),
        pitch=math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2])),
        yaw=math.atan2(rotation[1, 0], rotation[0, 0]),
    )


def _rotation(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation by |v| about v, by Rodrigues' formula."""
    x, y, z = vector
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return np.eye(3) + cross
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1 - math.cos(angle)) / (angle * angle) * cross @ cross
    )


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    transform = np.eye(4)
    cos, sin = math.cos(angle), math.sin(angle)
    transform[:2, :2] = ((cos, -sin), (sin, cos))
    return transform
