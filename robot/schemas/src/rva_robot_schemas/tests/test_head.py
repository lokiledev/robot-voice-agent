from rva_robot_schemas.head import HeadPose


def test_head_pose_reads_its_json() -> None:
    json = '{"x":0.01,"y":0.0,"z":0.15,"roll":0.0,"pitch":0.2,"yaw":-0.5}'

    assert HeadPose.model_validate_json(json) == HeadPose(
        x=0.01, y=0.0, z=0.15, roll=0.0, pitch=0.2, yaw=-0.5
    )
