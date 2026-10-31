from uuid import UUID


class VlaBusyError(Exception):
    """The VLA refused a task because another one is running."""

    def __init__(self, running_task_id: UUID) -> None:
        super().__init__(f"Task {running_task_id} is still running")
        self.running_task_id = running_task_id


class UnknownTaskError(Exception):
    def __init__(self, task_id: UUID) -> None:
        super().__init__(f"No task {task_id}")
        self.task_id = task_id
