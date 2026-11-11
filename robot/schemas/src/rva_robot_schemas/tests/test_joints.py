from rva_robot_schemas.joints import JointCommand, JointReading, JointState, JointTarget


def test_joint_state_reads_its_json() -> None:
    json = (
        '{"timestamp":12.5,"joints":['
        '{"name":"yaw_body","position":0.25,"velocity":-0.5},'
        '{"name":"left_antenna","position":0.0,"velocity":0.0}]}'
    )

    assert JointState.model_validate_json(json) == JointState(
        timestamp=12.5,
        joints=(
            JointReading(name="yaw_body", position=0.25, velocity=-0.5),
            JointReading(name="left_antenna", position=0.0, velocity=0.0),
        ),
    )


def test_joint_command_reads_its_json() -> None:
    json = '{"targets":[{"name":"stewart_1","position":0.3}]}'

    assert JointCommand.model_validate_json(json) == JointCommand(
        targets=(JointTarget(name="stewart_1", position=0.3),)
    )
