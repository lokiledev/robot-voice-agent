from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from rva_pydantic import RVABaseModel


class TaskRequest(RVABaseModel):
    instruction: str
    """Short-horizon natural-language command, e.g. "pick up the red cup"."""


class TaskRunning(RVABaseModel):
    state: Literal["running"] = "running"


class TaskSucceeded(RVABaseModel):
    state: Literal["succeeded"] = "succeeded"


class TaskFailed(RVABaseModel):
    state: Literal["failed"] = "failed"
    reason: str
    """Why the VLA gave up, in words the agent can relay to the user."""


class TaskCancelled(RVABaseModel):
    state: Literal["cancelled"] = "cancelled"


type TaskStatus = Annotated[
    TaskRunning | TaskSucceeded | TaskFailed | TaskCancelled, Field(discriminator="state")
]
type FinalTaskStatus = TaskSucceeded | TaskFailed | TaskCancelled


class Task(RVABaseModel):
    id: UUID
    instruction: str
    status: TaskStatus


class TaskEvent(RVABaseModel):
    """A status change of a task, pushed by the VLA."""

    task_id: UUID
    status: TaskStatus
    timestamp: datetime
