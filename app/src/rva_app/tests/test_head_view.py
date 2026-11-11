import math
from pathlib import Path

import numpy as np

from rva_app.head_view import pose_matrix, read_meshes
from rva_robot_schemas.head import HeadPose

# Reachy Mini's MJCF, cut down to a mesh on each part, the head's site and an antenna.
MJCF = """
<mujoco>
  <compiler meshdir="assets"/>
  <asset>
    <mesh file="foot.stl"/>
    <mesh file="shell.stl"/>
    <mesh file="face.stl"/>
    <mesh file="antenna.stl"/>
    <material name="white" rgba="1 1 1 1"/>
  </asset>
  <worldbody>
    <body name="body_foot_3dprint">
      <geom type="mesh" class="visual" mesh="foot" material="white"/>
      <body name="body_down_3dprint" pos="0 0 0.0348">
        <joint name="yaw_body"/>
        <geom type="mesh" class="visual" pos="0 0 0.1" mesh="shell"/>
        <geom type="mesh" class="collision" mesh="shell"/>
        <body name="stewart_rod">
          <body name="xl_330" pos="0 0 0.1">
            <geom type="mesh" class="visual" mesh="face" material="white"/>
            <site name="head" pos="0.01 0 0.02"/>
            <body name="antenna" pos="0 0.03 0.05">
              <joint name="left_antenna"/>
              <geom type="mesh" class="visual" pos="0 0 0.01" mesh="antenna"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_meshes_are_grouped_by_the_part_they_move_with(tmp_path: Path) -> None:
    (tmp_path / "reachy_mini.xml").write_text(MJCF)

    robot = read_meshes(tmp_path / "reachy_mini.xml")

    assets = tmp_path / "assets"
    assert [(mesh.part, mesh.file, mesh.rgba) for mesh in robot.meshes] == [
        ("base", assets / "foot.stl", (1.0, 1.0, 1.0, 1.0)),
        ("body", assets / "shell.stl", (0.7, 0.7, 0.7, 1.0)),
        ("head", assets / "face.stl", (1.0, 1.0, 1.0, 1.0)),
        # Carried by the head, at its joint's zero.
        ("head", assets / "antenna.stl", (0.7, 0.7, 0.7, 1.0)),
    ]
    np.testing.assert_allclose(robot.meshes[3].pose[:3, 3], (0.0, 0.03, 0.06))
    np.testing.assert_allclose(robot.body[:3, 3], (0.0, 0.0, 0.0348))
    np.testing.assert_allclose(robot.head_site[:3, 3], (0.01, 0.0, 0.02))


def test_a_head_pose_turns_by_yaw_then_pitch_then_roll() -> None:
    pose = HeadPose(x=0.0, y=0.0, z=0.15, roll=0.0, pitch=math.radians(90), yaw=math.radians(90))

    matrix = pose_matrix(pose)

    # Turned left a quarter, then looking down: its x axis points at the floor.
    np.testing.assert_allclose(matrix[:3, 0], (0.0, 0.0, -1.0), atol=1e-12)
    np.testing.assert_allclose(matrix[:3, 1], (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(matrix[:3, 3], (0.0, 0.0, 0.15))
