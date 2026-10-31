from datetime import UTC, datetime
from uuid import UUID

from rva_vla_schemas.task import TaskEvent, TaskFailed


def test_task_event_deserializes_failure_reason() -> None:
    serialized = (
        '{"task_id": "6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11",'
        ' "status": {"state": "failed", "cause": "error", "reason": "cup out of reach"},'
        ' "timestamp": "2026-10-31T10:00:00Z"}'
    )

    assert TaskEvent.model_validate_json(serialized) == TaskEvent(
        task_id=UUID("6f1c1f2e-8a51-4c43-9d43-0f6b1d0c9b11"),
        status=TaskFailed(cause="error", reason="cup out of reach"),
        timestamp=datetime(2026, 10, 31, 10, tzinfo=UTC),
    )
