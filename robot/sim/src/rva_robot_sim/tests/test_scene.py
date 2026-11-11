"""The real Reachy Mini in Genesis: needs its description, and a minute to compile the kernels."""

from pathlib import Path

import genesis as gs
import httpx
import numpy as np
import pytest
from genesis.constants import backend

from rva_pydantic import load_yaml
from rva_robot_schemas.joints import JointCommand, JointTarget
from rva_robot_sim.config import Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.scene import ReachyMiniScene, UnknownJointError

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"


@pytest.fixture(scope="module")
def scene() -> ReachyMiniScene:
    config = load_yaml(SimConfig, SIM_YAML)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
    gs.init(backend=backend.cpu, logging_level="warning")
    # Without people, which would only take downloads and Blender, and rasterized, which needs no
    # ray-tracing GPU.
    return ReachyMiniScene(
        mjcf, config.table_height, (), Rasterizer(), None, config.physics_dt, viewer=False
    )


@pytest.mark.download
@pytest.mark.timeout(600)
def test_joints_go_where_they_are_told_and_the_camera_sees(scene: ReachyMiniScene) -> None:
    # Odd and even motors of the Stewart platform turn in opposite senses to lift the head.
    lift = (("stewart_1", 0.2), ("stewart_2", -0.2), ("stewart_3", 0.2))
    lift += (("stewart_4", -0.2), ("stewart_5", 0.2), ("stewart_6", -0.2))
    targets = (("yaw_body", 0.4), *lift)
    scene.command(JointCommand(targets=tuple(JointTarget(name=n, position=p) for n, p in targets)))
    for _ in range(400):
        scene.step()

    state = scene.joint_state(timestamp=7.0)
    frame = scene.render(timestamp=7.0)

    assert state.timestamp == 7.0
    positions = {joint.name: joint.position for joint in state.joints}
    assert list(positions) == [
        "yaw_body",
        *(f"stewart_{motor}" for motor in range(1, 7)),
        "right_antenna",
        "left_antenna",
    ]
    for name, position in targets:
        assert positions[name] == pytest.approx(position, abs=0.02)
    assert frame.image.shape == (720, 1280, 3)
    # Not a blank image: the sky and the floor are seen.
    assert np.ptp(frame.image) > 50


@pytest.mark.download
@pytest.mark.timeout(600)
def test_a_command_naming_an_unknown_joint_is_refused_whole(scene: ReachyMiniScene) -> None:
    command = JointCommand(
        targets=(
            JointTarget(name="right_antenna", position=1.0),
            JointTarget(name="neck_pitch", position=0.1),
        )
    )

    with pytest.raises(UnknownJointError, match=r"No actuated joints \['neck_pitch'\]"):
        scene.command(command)
    for _ in range(200):
        scene.step()

    antenna = next(j for j in scene.joint_state(timestamp=0.0).joints if j.name == "right_antenna")
    assert antenna.position == pytest.approx(0.0, abs=0.02)
