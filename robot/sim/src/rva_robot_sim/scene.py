import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, assert_never

import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
import numpy as np
import numpy.typing as npt
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import R_to_quat, quat_to_R
from genesis.utils.misc import tensor_to_array
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

from rva_robot_schemas.camera import CameraFrame
from rva_robot_schemas.joints import JointCommand, JointReading, JointState
from rva_robot_schemas.sound import SoundDirection
from rva_robot_sim import nyx_skinned
from rva_robot_sim.actor import Actor, vertex_order
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer
from rva_robot_sim.hearing import heard_azimuth
from rva_robot_sim.mjcf import CameraMount, read_facts
from rva_robot_sim.npc import PersonPose
from rva_robot_sim.people import standing
from rva_robot_sim.scenes import Interior, Prop
from rva_robot_sim.servos import PassiveJointsConfig, ServoConfig, Servos
from rva_robot_sim.skinning import Floats

_TABLE_SIZE: Final = (0.6, 1.0)
"""Depth and width of the table top, in meters."""
_TABLE_X: Final = -0.15
"""The table's center, behind the robot, which faces +x near the front edge."""
_TABLE_COLOR: Final = (0.55, 0.4, 0.28)
_FLOOR_COLOR: Final = (0.42, 0.4, 0.37)
"""A grey concrete, rather than Genesis' checkerboard."""
_SUN: Final[Mapping[str, object]] = {
    "type": "directional",
    "dir": (-0.3, 0.2, -0.9),
    "intensity": 3.0,
    "shadow": True,
}
"""High, from behind the robot on its left, for Nyx, where no room keeps it out."""
_SPEAKING: Final = 0.1
"""How loud a person must be, as their mouth opens from 0 to 1, to be heard speaking."""
_NEAR_M: Final = 0.02
_FAR_M: Final = 50.0


class UnknownJointError(ValueError):
    pass


class UnknownPersonError(ValueError):
    pass


class ReachyMiniScene:
    """A Reachy Mini on a table, with people standing around, in Genesis, with its head camera.

    The joints are position controlled, as on the robot. Genesis must be initialized before.
    """

    def __init__(
        self,
        mjcf: Path,
        table_height: float,
        interior: tuple[Path, Interior] | None,
        people: Mapping[str, tuple[Path, Actor]],
        carried: Mapping[str, tuple[Path, Prop]],
        servos: Sequence[ServoConfig],
        passive: PassiveJointsConfig,
        renderer: CameraRenderer,
        hdri: Path | None,
        physics_dt: float,
        viewer: bool,
    ) -> None:
        """Build the scene, which compiles Genesis' kernels: a minute the first time.

        Args:
            mjcf: Reachy Mini's MJCF.
            table_height: Of the table top the robot stands on, in meters.
            interior: The room around it and its model, from `install_model`, whose furniture
                the robot then stands on; None, and it stands on a table on a bare floor.
            people: Each person's avatar and the actor that plays it, from `actor.cast`, by their
                name.
            carried: What some of the people carry and its model, from `install_model`, by
                the name of who carries it.
            servos: What drives each actuated joint, in place of the MJCF's actuators.
            passive: The joints no servo drives, in place of the MJCF's joints' damping and
                friction.
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
            # The robot's parts collide as the convex hulls of their meshes, much bigger than the
            # body's shell: the head would bump into hulls that are not there. The table and the
            # floor still hold it.
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -0.6, table_height + 0.5),
                camera_lookat=(0.0, 0.0, table_height + 0.15),
                camera_fov=40,
                # The caller paces the steps to real time: the viewer pacing them too would add
                # its own waits to the loop's.
                realtime_factor=None,
            ),
        )
        if interior is None:
            self._scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(color=_FLOOR_COLOR))
        else:
            self._scene.add_entity(_room(*interior))
        self._scene.add_entity(
            gs.morphs.Box(
                size=(*_TABLE_SIZE, table_height),
                pos=(_TABLE_X, 0.0, table_height / 2),
                fixed=True,
                # Inside the room's furniture, which is only seen: this holds the robot.
                visualization=interior is None,
            ),
            surface=gs.surfaces.Default(color=_TABLE_COLOR),
        )
        # Seen, not touched: the robot cannot reach them. Each mesh stays at the origin, in its
        # rest pose, and `animate` moves its vertices.
        self._people: Final = {
            name: (
                self._scene.add_entity(
                    gs.morphs.Mesh(
                        file=str(avatar), enable_custom_vverts=True, fixed=True, collision=False
                    )
                ),
                actor,
            )
            for name, (avatar, actor) in people.items()
        }
        # Seen, not touched either: `animate` moves each as a whole with who carries it.
        props = {
            name: self._scene.add_entity(
                gs.morphs.Mesh(file=str(model), scale=prop.scale, fixed=True, collision=False)
            )
            for name, (model, prop) in carried.items()
        }
        self._robot: Final = self._scene.add_entity(
            gs.morphs.MJCF(file=str(mjcf), pos=(0.0, 0.0, table_height))
        )
        mount = facts.head_camera
        self._mount: Final = mount
        # Each person's loudness and head, as last animated.
        self._voices: list[tuple[float, Floats]] = []
        if isinstance(renderer, NyxRenderer):
            # Nyx would show the people still, in their rest pose.
            nyx_skinned.install()
        # A sensor fixed to the head's link, which renders when read rather than at every step.
        lights = [_SUN] if interior is None else _lamps(interior[1])
        self._camera: Final = self._scene.add_sensor(
            _camera(renderer, hdri, lights, mount, self._robot)
        )
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
        self._servos: Final = Servos(servos, self._names)
        # The MJCF's actuators are ideal, 20 N m each: the servos' torques replace them, and
        # their gearboxes the joints' inertia and friction.
        self._robot.set_dofs_armature(self._servos.armature, self._dofs)
        self._robot.set_dofs_frictionloss(self._servos.frictionloss, self._dofs)
        self._robot.set_dofs_damping(np.zeros(len(self._dofs)), self._dofs)
        # The MJCF gives the rods' ball joints an actuator's damping and friction, which only
        # 20 N m actuators could overcome.
        rods = [
            dof for name in passive.joints for dof in self._robot.get_joint(name).dofs_idx_local
        ]
        self._robot.set_dofs_damping(np.zeros(len(rods)), rods)
        self._robot.set_dofs_frictionloss(np.full(len(rods), passive.frictionloss_nm), rods)
        self._robot.set_dofs_armature(np.full(len(rods), passive.armature), rods)
        self._orders: Final = [
            vertex_order(actor.skin, tensor_to_array(entity.get_vverts()))
            for entity, actor in self._people.values()
        ]
        # Each prop with the middle of its bottom, from its origin.
        self._carried: Final = {
            name: (entity, _bottom(tensor_to_array(entity.get_vAABB())))
            for name, entity in props.items()
        }
        self.animate(0.0)

    def step(self) -> None:
        """One physics step, each joint driven by its servo towards its goal."""
        positions = self._robot.get_dofs_position(self._dofs).numpy()
        velocities = self._robot.get_dofs_velocity(self._dofs).numpy()
        torques = self._servos.torques(self._targets, positions, velocities)
        self._robot.control_dofs_force(torques, self._dofs)
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

    def place(self, pose: PersonPose) -> None:
        """Move one of the people, from the next `animate`.

        Raises:
            UnknownPersonError: If there is no such person.
        """
        if pose.person not in self._people:
            raise UnknownPersonError(
                f"Nobody named {pose.person!r}; there are {sorted(self._people)}"
            )
        _, actor = self._people[pose.person]
        actor.place(standing(pose.position, pose.turned_deg))

    def animate(self, time: float) -> None:
        """Pose the people as they are `time` seconds into the simulation; before a render."""
        voices = []
        for (name, (entity, actor)), order in zip(self._people.items(), self._orders, strict=True):
            bones = actor.pose(time)
            entity.set_vverts(actor.vertices(bones, time)[order])
            voices.append((actor.voice(time), actor.head(bones)))
            if name in self._carried:
                prop, bottom = self._carried[name]
                frame = actor.carrying(bones)
                prop.set_pos(frame[:3, 3] - frame[:3, :3] @ bottom)
                prop.set_quat(R_to_quat(frame[:3, :3]))
        self._voices = voices

    def sound_direction(self, timestamp: float) -> SoundDirection:
        """Where the microphones hear speech from, as the people were last animated.

        From the loudest person speaking, at their head; none when nobody speaks.
        """
        speaking = [(loudness, head) for loudness, head in self._voices if loudness > _SPEAKING]
        if not speaking:
            return SoundDirection(timestamp=timestamp, speech_azimuth=None)
        _, head = max(speaking, key=lambda voice: voice[0])
        camera = self.link_pose(self._mount.body) @ _transform(self._mount)
        return SoundDirection(timestamp=timestamp, speech_azimuth=heard_azimuth(camera, head))

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

    def link_pose(self, name: str) -> npt.NDArray[np.float64]:
        """The 4×4 pose of the robot's body `name` in the world, whose origin is under the robot."""
        link = self._robot.get_link(name)
        pose = np.eye(4)
        pose[:3, :3] = quat_to_R(link.get_quat().numpy())
        pose[:3, 3] = link.get_pos().numpy()
        return pose

    def render(self, timestamp: float) -> CameraFrame:
        """What the head camera sees now."""
        rgb = tensor_to_array(self._camera.read().rgb)
        return CameraFrame(timestamp=timestamp, image=np.ascontiguousarray(rgb[..., ::-1]))


def _bottom(box: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """3: the middle of the bottom of a box given as its lowest and highest corners."""
    low, high = box
    return np.array(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]))


def _room(model: Path, interior: Interior) -> gs.morphs.Mesh:
    """The room's model, turned and moved so that the robot stands where the scene puts it."""
    turn = math.radians(-interior.robot_faces_deg)
    x, y = interior.robot_at
    return gs.morphs.Mesh(
        file=str(model),
        scale=interior.scale,
        pos=(
            -(x * math.cos(turn) - y * math.sin(turn)),
            -(x * math.sin(turn) + y * math.cos(turn)),
            0.0,
        ),
        euler=(0.0, 0.0, -interior.robot_faces_deg),
        fixed=True,
        collision=False,
    )


def _lamps(interior: Interior) -> list[Mapping[str, object]]:
    return [
        {"type": "point", "pos": lamp, "intensity": interior.lamp_lumens, "shadow": True}
        for lamp in interior.lamps
    ]


def _camera(
    renderer: CameraRenderer,
    hdri: Path | None,
    lights: Sequence[Mapping[str, object]],
    mount: CameraMount,
    robot: RigidEntity,
) -> NyxCameraOptions | gs.sensors.RasterizerCameraOptions:
    link = robot.get_link(mount.body).idx_local
    match renderer:
        case NyxRenderer():
            assert hdri is not None, "installed for Nyx"
            sky = environment_map(renderer, hdri)
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
                lights=[dict(light) for light in lights],
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


def environment_map(renderer: NyxRenderer, hdri: Path) -> nps.EnvironmentMapAsset:
    """The panorama Nyx shows around the world and lights it with, from `install_hdri`."""
    sky = nps.EnvironmentMapAsset()
    sky.texture = str(hdri)
    sky.layout = nps.EEnvMapLayout.LongLat
    sky.multiplier = renderer.environment.brightness
    sky.rotation = renderer.environment.rotation_deg
    return sky


def _transform(mount: CameraMount) -> npt.NDArray[np.float64]:
    """The camera's pose in its body. Genesis' cameras look along -z with +y up, like MuJoCo's."""
    transform = np.eye(4)
    transform[:3, :3] = quat_to_R(np.array(mount.quaternion))
    transform[:3, 3] = mount.position
    return transform
