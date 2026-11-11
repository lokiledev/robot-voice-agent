from pathlib import Path
from typing import Final

import genesis as gs
import numpy as np
import numpy.typing as npt
from genesis.utils.geom import quat_to_R

from rva_robot_schemas.camera import CameraFrame
from rva_robot_schemas.joints import JointCommand, JointReading, JointState
from rva_robot_sim.mjcf import CameraMount, read_facts

_TABLE_SIZE: Final = (0.6, 1.0)
"""Depth and width of the table top, in meters."""
_TABLE_X: Final = -0.15
"""The table's center, behind the robot, which faces +x near the front edge."""
_TABLE_COLOR: Final = (0.55, 0.4, 0.28)


class UnknownJointError(ValueError):
    pass


class ReachyMiniScene:
    """A Reachy Mini on a table in an empty world, in Genesis, with its head camera.

    The joints are position controlled, as on the robot. Genesis must be initialized before.
    """

    def __init__(self, mjcf: Path, table_height: float, physics_dt: float, viewer: bool) -> None:
        """Build the scene, which compiles Genesis' kernels: a minute the first time.

        Args:
            mjcf: Reachy Mini's MJCF.
            table_height: Of the table top the robot stands on, in meters.
            physics_dt: Seconds of simulated time per step.
            viewer: Open Genesis' viewer, to watch the scene from outside.
        """
        facts = read_facts(mjcf)
        self._scene: Final = gs.Scene(
            show_viewer=viewer,
            sim_options=gs.options.SimOptions(dt=physics_dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -0.6, table_height + 0.5),
                camera_lookat=(0.0, 0.0, table_height + 0.15),
                camera_fov=40,
                # The caller paces the steps to real time: the viewer pacing them too would add
                # its own waits to the loop's.
                realtime_factor=None,
            ),
        )
        self._scene.add_entity(gs.morphs.Plane())
        self._scene.add_entity(
            gs.morphs.Box(
                size=(*_TABLE_SIZE, table_height),
                pos=(_TABLE_X, 0.0, table_height / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=_TABLE_COLOR),
        )
        self._robot: Final = self._scene.add_entity(
            gs.morphs.MJCF(file=str(mjcf), pos=(0.0, 0.0, table_height))
        )
        mount = facts.head_camera
        self._camera: Final = self._scene.add_camera(
            res=(mount.width, mount.height), fov=mount.fov_y, GUI=False
        )
        self.camera_size: Final = (mount.width, mount.height)
        """Width and height of the head camera's frames, the MJCF's."""
        self._scene.build()
        self._camera.attach(self._robot.get_link(mount.body), _transform(mount))
        self._names: Final = facts.actuated
        self._dofs: Final = [self._robot.get_joint(name).dofs_idx_local[0] for name in self._names]
        lower, upper = self._robot.get_dofs_limit(self._dofs)
        self._lower: Final = lower.numpy()
        self._upper: Final = upper.numpy()
        self._targets: Final = np.zeros(len(self._names))
        """Where each joint is driven to, starting at the MJCF's rest pose, head upright."""
        self._robot.control_dofs_position(self._targets, self._dofs)

    def step(self) -> None:
        self._scene.step()

    def command(self, command: JointCommand) -> None:
        """Drive the joints `command` names to their targets, clamped to their range.

        Raises:
            UnknownJointError: If a target names a joint that is not actuated; none is applied.
        """
        unknown = [target.name for target in command.targets if target.name not in self._names]
        if unknown:
            raise UnknownJointError(f"No actuated joints {unknown}; there are {self._names}")
        for target in command.targets:
            index = self._names.index(target.name)
            self._targets[index] = np.clip(target.position, self._lower[index], self._upper[index])
        self._robot.control_dofs_position(self._targets, self._dofs)

    def joint_state(self, timestamp: float) -> JointState:
        positions = self._robot.get_dofs_position(self._dofs).numpy()
        velocities = self._robot.get_dofs_velocity(self._dofs).numpy()
        return JointState(
            timestamp=timestamp,
            joints=tuple(
                JointReading(name=name, position=float(position), velocity=float(velocity))
                for name, position, velocity in zip(self._names, positions, velocities, strict=True)
            ),
        )

    def render(self, timestamp: float) -> CameraFrame:
        """What the head camera sees now."""
        self._camera.move_to_attach()
        rgb, *_ = self._camera.render()
        return CameraFrame(timestamp=timestamp, image=np.ascontiguousarray(rgb[..., ::-1]))


def _transform(mount: CameraMount) -> npt.NDArray[np.float64]:
    """The camera's pose in its body. Genesis' cameras look along -z with +y up, like MuJoCo's."""
    transform = np.eye(4)
    transform[:3, :3] = quat_to_R(np.array(mount.quaternion))
    transform[:3, 3] = mount.position
    return transform
