from uuid import UUID

from rva_body_schemas.motion import GoToHuman, Motion, PlayAnimation, describe
from rva_vla_schemas.task import TaskRunning


def test_motion_deserializes_its_action_by_kind() -> None:
    serialized = (
        '{"id": "6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11",'
        ' "action": {"kind": "animation", "animation": "wave_right_hand"},'
        ' "status": {"state": "running"}}'
    )

    assert Motion.model_validate_json(serialized) == Motion(
        id=UUID("6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11"),
        action=PlayAnimation(animation="wave_right_hand"),
        status=TaskRunning(),
    )


def test_describe_motions() -> None:
    assert describe(PlayAnimation(animation="dance")) == "play the dance animation"
    assert describe(GoToHuman(user_id=UUID(int=1))) == (
        "go to user 00000000-0000-0000-0000-000000000001"
    )
