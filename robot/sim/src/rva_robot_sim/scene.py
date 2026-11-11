from collections.abc import Sequence
from pathlib import Path
from typing import Final, assert_never

import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
import numpy as np
import numpy.typing as npt
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import quat_to_R
from genesis.utils.misc import tensor_to_array
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from rva_robot_schemas.camera import CameraFrame
from rva_robot_schemas.joints import JointCommand, JointReading, JointState
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer
from rva_robot_sim.mjcf import CameraMount, read_facts
from rva_robot_sim.people import PlacedPerson, yaw_deg

_TABLE_SIZE: Final = (0.6, 1.0)
"""Depth and width of the table top, in meters."""
_TABLE_X: Final = -0.15
"""The table's center, behind the robot, which faces +x near the front edge."""
_TABLE_COLOR: Final = (0.55, 0.4, 0.28)
_FLOOR_COLOR: Final = (0.42, 0.4, 0.37)
"""A grey concrete, rather than Genesis' checkerboard."""
_SUN: Final = {"type": "directional", "dir": (-0.3, 0.2, -0.9), "intensity": 3.0, "shadow": True}
"""High, from behind the robot on its left, for Nyx."""
_NEAR_M: Final = 0.02
_FAR_M: Final = 50.0


class UnknownJointError(ValueError):
    pass


class ReachyMiniScene:
    """A Reachy Mini on a table, with people standing around, in Genesis, with its head camera.

    The joints are position controlled, as on the robot. Genesis must be initialized before.
    """

    def __init__(
        self,
        mjcf: Path,
        table_height: float,
        people: Sequence[tuple[Path, PlacedPerson]],
        renderer: CameraRenderer,
        hdri: Path | None,
        physics_dt: float,
        viewer: bool,
    ) -> None:
        """Build the scene, which compiles Genesis' kernels: a minute the first time.

        Args:
            mjcf: Reachy Mini's MJCF.
            table_height: Of the table top the robot stands on, in meters.
            people: Each person's baked avatar, from `install_avatar`, and where they stand.
            renderer: What renders the head camera.
            hdri: The panorama Nyx lights the world with, from `install_hdri`; `None` with the
                rasterizer, which takes none.
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
        self._scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=_FLOOR_COLOR))
        self._scene.add_entity(
            gs.morphs.Box(
                size=(*_TABLE_SIZE, table_height),
                pos=(_TABLE_X, 0.0, table_height / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=_TABLE_COLOR),
        )
        for avatar, person in people:
            # Seen, not touched: the robot cannot reach them.
            self._scene.add_entity(
                gs.morphs.Mesh(
                    file=str(avatar),
                    pos=(*person.position, 0.0),
                    euler=(0.0, 0.0, yaw_deg(person)),
                    fixed=True,
                    collision=False,
                )
            )
        self._robot: Final = self._scene.add_entity(
            gs.morphs.MJCF(file=str(mjcf), pos=(0.0, 0.0, table_height))
        )
        mount = facts.head_camera
        # A sensor fixed to the head's link, which renders when read rather than at every step.
        self._camera: Final = self._scene.add_sensor(_camera(renderer, hdri, mount, self._robot))
        self.camera_size: Final = (mount.width, mount.height)
        """Width and height of the head camera's frames, the MJCF's."""
        self._scene.build()
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
        rgb = tensor_to_array(self._camera.read().rgb)
        return CameraFrame(timestamp=timestamp, image=np.ascontiguousarray(rgb[..., ::-1]))


def _camera(
    renderer: CameraRenderer, hdri: Path | None, mount: CameraMount, robot: RigidEntity
) -> NyxCameraOptions | gs.sensors.RasterizerCameraOptions:
    link = robot.get_link(mount.body).idx_local
    match renderer:
        case NyxRenderer():
            assert hdri is not None, "installed for Nyx"
            sky = nps.EnvironmentMapAsset()
            sky.texture = str(hdri)
            sky.layout = nps.EEnvMapLayout.LongLat
            sky.multiplier = renderer.environment.brightness
            sky.rotation = renderer.environment.rotation_deg
            return NyxCameraOptions(
                res=(mount.width, mount.height),
                fov=mount.fov_y,
                near=_NEAR_M,
                far=_FAR_M,
                entity_idx=robot.idx,
                link_idx_local=link,
                offset_T=_transform(mount),
                spp=renderer.samples_per_pixel,
                render_mode=npr.ERenderMode.FastPathTracer,
                env_maps=(sky,),
                lights=[_SUN],
            )
        case Rasterizer():
            return gs.sensors.RasterizerCameraOptions(
                res=(mount.width, mount.height),
                fov=mount.fov_y,
                near=_NEAR_M,
                far=_FAR_M,
                entity_idx=robot.idx,
                link_idx_local=link,
                offset_T=_transform(mount),
            )
        case _:
            assert_never(renderer)


def _transform(mount: CameraMount) -> npt.NDArray[np.float64]:
    """The camera's pose in its body. Genesis' cameras look along -z with +y up, like MuJoCo's."""
    transform = np.eye(4)
    transform[:3, :3] = quat_to_R(np.array(mount.quaternion))
    transform[:3, 3] = mount.position
    return transform
