from rva_robot_schemas.head import GazeTarget, HeadPose


def test_head_pose_reads_its_json() -> None:
    json = '{"x":0.01,"y":0.0,"z":0.15,"roll":0.0,"pitch":0.2,"yaw":-0.5}'

    assert HeadPose.model_validate_json(json) == HeadPose(
        x=0.01, y=0.0, z=0.15, roll=0.0, pitch=0.2, yaw=-0.5
    )


def test_gaze_target_reads_its_json() -> None:
    json = '{"timestamp":12.5,"x":-0.1,"y":0.05,"z":1.2}'

    assert GazeTarget.model_validate_json(json) == GazeTarget(timestamp=12.5, x=-0.1, y=0.05, z=1.2)
