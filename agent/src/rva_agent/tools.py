import json
from typing import Final

import pydantic

from rva_agent.permission import ActionPermission
from rva_agent.tasks import TaskSupervisor
from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import ToolCall, ToolSpec
from rva_vla_schemas.errors import VlaBusyError


class StartTaskArgs(RVABaseModel):
    instruction: str
    """Short command for the body, phrased like the skill examples."""


class NoArgs(RVABaseModel):
    pass


class ToolResult(RVABaseModel):
    content: str
    """What the LLM reads."""
    needs_reply: bool
    """Whether the LLM should speak about it. A task that just started needs no comment: the LLM
    announced it in the same message, and its end arrives later as a robot event."""


_SPECS: Final = (
    ToolSpec(
        name="start_task",
        description="Start a short physical task. Returns as soon as it begins; its end arrives later as a robot event.",
        parameters=StartTaskArgs.model_json_schema(),
    ),
    ToolSpec(
        name="cancel_task",
        description="Stop the task that is running.",
        parameters=NoArgs.model_json_schema(),
    ),
    ToolSpec(
        name="get_task_status",
        description="Tell which task is running, if any.",
        parameters=NoArgs.model_json_schema(),
    ),
)


class TaskTools:
    """The tools the LLM uses to drive the robot's body."""

    specs: Final = _SPECS

    def __init__(self, supervisor: TaskSupervisor, permission: ActionPermission) -> None:
        self._supervisor: Final = supervisor
        self._permission: Final = permission

    async def call(self, call: ToolCall) -> ToolResult:
        """Run a tool call and return its result for the LLM.

        Mistakes of the LLM (unknown tool, bad arguments) are returned as results so it can correct
        itself.
        """
        try:
            match call.name:
                case "start_task":
                    return await self._start(StartTaskArgs.model_validate_json(call.arguments))
                case "cancel_task":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    return await self._cancel()
                case "get_task_status":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    return self._status()
                case _:
                    return _result(error=f"There is no tool named {call.name}.")
        except pydantic.ValidationError as error:
            return _result(error=f"Invalid arguments: {error}")

    async def _start(self, args: StartTaskArgs) -> ToolResult:
        if (refusal := self._permission.refusal()) is not None:
            return _result(
                error=f"{refusal} Only operators can ask you to move."
                " Tell them kindly that you can talk with them but not do this."
            )
        try:
            task = await self._supervisor.start(args.instruction)
        except VlaBusyError:
            return _result(
                error=f"Busy with {self._running()}. Do not retry: cancel it, or wait for its robot_event."
            )
        started = _result(
            status="started",
            task=task.instruction,
            next="It is running now. Its result arrives later as a robot_event; until then do not start another task or say it is done.",
        )
        return ToolResult(content=started.content, needs_reply=False)

    async def _cancel(self) -> ToolResult:
        task = await self._supervisor.cancel()
        if task is None:
            return _result(error="No task is running.")
        return _result(status="cancelling", task=task.instruction)

    def _status(self) -> ToolResult:
        if self._supervisor.active is None:
            return _result(status="idle")
        return _result(status="running", task=self._supervisor.active.instruction)

    def _running(self) -> str:
        active = self._supervisor.active
        return "another task" if active is None else repr(active.instruction)


def _result(**fields: str) -> ToolResult:
    return ToolResult(content=json.dumps(fields), needs_reply=True)
