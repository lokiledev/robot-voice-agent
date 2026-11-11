from typing import Final

from rva_robot_kinematics.head import HeadKinematics
from rva_robot_schemas.head import HeadPose
from rva_robot_schemas.joints import JointCommand


class HeadController:
    """Puts the head where it is told, turning the body only when the head needs it.

    The body keeps the yaw last commanded, rather than the one measured, so that its lag does not
    pull it along.
    """

    def __init__(self, kinematics: HeadKinematics, body_yaw: float) -> None:
        """Start from the body's yaw now.

        Args:
            kinematics: Of the robot's head.
            body_yaw: Where the body is turned now, in radians.
        """
        self._kinematics: Final = kinematics
        self._body_yaw = body_yaw

    def aim(self, target: HeadPose) -> JointCommand:
        """The joints' targets that put the head at `target`.

        Raises:
            UnreachableHeadPoseError: If the head cannot reach `target`; nothing changes then.
        """
        command = self._kinematics.solve(target, self._body_yaw)
        body, *_ = command.targets
        self._body_yaw = body.position
        return command
