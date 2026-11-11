from pathlib import Path

import pytest

from rva_robot_sim.mjcf import CameraMount, RobotFacts, read_facts

# Reachy Mini's MJCF, cut down to what is read from it.
MJCF = """
<mujoco model="reachy_mini">
  <worldbody>
    <camera name="studio_close" pos="0.8 -0.2 0.4"/>
    <body name="body_foot_3dprint">
      <body name="xl_330" pos="-0.085 0 0" quat="0 1 0 0">
        <body pos="-0.03 -0.05 0.002" quat="0.5 -0.5 0.5 0.5">
          <camera name="eye_camera" mode="fixed" resolution="1280 720" fovy="80"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="yaw_body" joint="yaw_body"/>
    <position name="stewart_1" joint="stewart_1"/>
    <position name="left_antenna" joint="left_antenna"/>
  </actuator>
</mujoco>
"""


def test_reads_the_actuated_joints_and_the_head_camera(tmp_path: Path) -> None:
    path = tmp_path / "reachy_mini.xml"
    path.write_text(MJCF)

    assert read_facts(path) == RobotFacts(
        actuated=("yaw_body", "stewart_1", "left_antenna"),
        head_camera=CameraMount(
            body="xl_330",
            position=(-0.03, -0.05, 0.002),
            quaternion=(0.5, -0.5, 0.5, 0.5),
            width=1280,
            height=720,
            fov_y=80.0,
        ),
    )


def test_a_camera_with_an_offset_of_its_own_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "reachy_mini.xml"
    path.write_text(MJCF.replace('mode="fixed"', 'pos="0 0 0.01"'))

    with pytest.raises(ValueError, match="eye_camera has an offset of its own"):
        read_facts(path)
