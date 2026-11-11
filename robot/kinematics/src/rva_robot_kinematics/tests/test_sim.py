"""The simulated Reachy Mini: needs its description, and a minute to compile Genesis' kernels."""

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import genesis as gs
import httpx
import numpy as np
import numpy.typing as npt
import pytest
from genesis.constants import backend
from genesis.utils.geom import quat_to_R

from rva_pydantic import load_yaml
from rva_robot_kinematics.geometry import HEAD_SITE, HeadGeometry, read_head_geometry
from rva_robot_kinematics.head import HeadKinematics, HeadPose
from rva_robot_schemas.joints import JointCommand
from rva_robot_sim.config import Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.scene import ReachyMiniScene

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"
NEUTRAL = 0.177
"""The head's height in Pollen's SDK when upright."""
HEAD_LINK = "xl_330"
"""The body the head site is on."""


class _Robot:
    def __init__(self, mjcf: Path, config: SimConfig) -> None:
        # Without people or Nyx: only the robot's joints are looked at.
        self.scene = ReachyMiniScene(
            mjcf, config.table_height, (), Rasterizer(), None, config.physics_dt, viewer=False
        )
        self.geometry = read_head_geometry(mjcf)
        self.table_height = config.table_height
        self.head_in_link = _head_in_link(mjcf)


@pytest.fixture(scope="module")
def robot() -> Iterator[_Robot]:
    config = load_yaml(SimConfig, SIM_YAML)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
    gs.init(backend=backend.cpu, logging_level="warning")
    yield _Robot(mjcf, config)
    gs.destroy()


def _pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> HeadPose:
    """A pose with its angles in degrees."""
    return HeadPose(
        x=x,
        y=y,
        z=z,
        roll=math.radians(roll),
        pitch=math.radians(pitch),
        yaw=math.radians(yaw),
    )


@pytest.mark.download
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("target", "body_yaw", "expected_body_yaw"),
    [
        pytest.param(_pose(0.0, 0.0, NEUTRAL, 0.0, 0.0, 0.0), 0.0, 0.0, id="neutral"),
        pytest.param(_pose(0.0, 0.0, NEUTRAL, 0.0, 15.0, 0.0), 0.0, 0.0, id="looking down"),
        pytest.param(_pose(0.0, 0.0, NEUTRAL, 10.0, 0.0, 0.0), 0.0, 0.0, id="tilted"),
        pytest.param(_pose(0.0, 0.0, NEUTRAL, 0.0, 0.0, 30.0), 0.0, 0.0, id="turned left"),
        pytest.param(_pose(0.0, 0.0, NEUTRAL + 0.01, 0.0, 0.0, 0.0), 0.0, 0.0, id="lifted"),
        pytest.param(
            _pose(0.005, -0.004, 0.18, 5.0, -10.0, 20.0), -10.0, -10.0, id="all, body turned"
        ),
        pytest.param(_pose(0.0, 0.0, NEUTRAL, 0.0, 0.0, 75.0), 0.0, 45.0, id="body following"),
    ],
)
def test_the_head_goes_where_asked(
    robot: _Robot, target: HeadPose, body_yaw: float, expected_body_yaw: float
) -> None:
    # The platform's hold on the head's yaw weakens as it turns the head far from the body: at
    # Pollen's 65°, the motors' small errors turn it by several degrees.
    kinematics = HeadKinematics(robot.geometry, max_head_yaw=math.radians(30))
    command = kinematics.solve(target, math.radians(body_yaw))

    robot.scene.command(command)
    for _ in range(400):
        robot.scene.step()

    head = _head_pose(robot)
    joints = {joint.name: joint.position for joint in robot.scene.joint_state(0.0).joints}
    assert _positions(command)["yaw_body"] == pytest.approx(math.radians(expected_body_yaw))
    assert (head.x, head.y, head.z) == pytest.approx((target.x, target.y, target.z), abs=0.002)
    assert (head.roll, head.pitch, head.yaw) == pytest.approx(
        (target.roll, target.pitch, target.yaw), abs=math.radians(1.5)
    )
    # The motors only get near their targets; the head is where the angles they reached put it.
    assert _positions(_still_body(robot.geometry).solve(head, joints["yaw_body"])) == pytest.approx(
        {name: joints[name] for name in _positions(command)}, abs=0.002
    )


def _still_body(geometry: HeadGeometry) -> HeadKinematics:
    """Kinematics that never turn the body: the platform turns the head any way it can."""
    return HeadKinematics(geometry, max_head_yaw=math.pi)


def _positions(command: JointCommand) -> dict[str, float]:
    return {target.name: target.position for target in command.targets}


def _head_pose(robot: _Robot) -> HeadPose:
    """The head's pose in the robot's base frame, as Genesis has it."""
    pose = robot.scene.link_pose(HEAD_LINK) @ robot.head_in_link
    rotation = pose[:3, :3]
    x, y, z = pose[:3, 3]
    return HeadPose(
        x=float(x),
        y=float(y),
        z=float(z) - robot.table_height,
        roll=math.atan2(rotation[2, 1], rotation[2, 2]),
        pitch=-math.asin(rotation[2, 0]),
        yaw=math.atan2(rotation[1, 0], rotation[0, 0]),
    )


def _head_in_link(mjcf: Path) -> npt.NDArray[np.float64]:
    """The pose of the head site in its body, from the MJCF."""
    site = next(s for s in ET.parse(mjcf).getroot().iter("site") if s.get("name") == HEAD_SITE)
    pose = np.eye(4)
    pose[:3, :3] = quat_to_R(np.array([float(value) for value in site.attrib["quat"].split()]))
    pose[:3, 3] = [float(value) for value in site.attrib["pos"].split()]
    return pose
