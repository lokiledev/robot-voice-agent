import json
from typing import Final

import pydantic

from rva_agent.permission import ActionPermission
from rva_agent.tools import ToolResult
from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import ToolCall, ToolSpec


class NoArgs(RVABaseModel):
    pass


_SPECS: Final = (
    ToolSpec(
        name="mute",
        description="Stop speaking until someone lets you talk again, e.g. when told to stop talking or be quiet. You keep listening and acting.",
        parameters=NoArgs.model_json_schema(),
    ),
    ToolSpec(
        name="unmute",
        description="Talk again after being muted, when told you can talk again.",
        parameters=NoArgs.model_json_schema(),
    ),
)


class MuteTools:
    """Whether the robot may speak, and the tools to change it.

    Anyone can mute the robot, like anyone can stop it. Letting it talk again needs an operator in
    front of it, so the person who silenced it is not overruled by a stranger.
    """

    specs: Final = _SPECS
    names: Final = frozenset(spec.name for spec in _SPECS)

    def __init__(self, permission: ActionPermission) -> None:
        self._permission: Final = permission
        self.muted = False

    async def call(self, call: ToolCall) -> ToolResult:
        """Run a tool call and return its result for the LLM; its mistakes are results too."""
        try:
            NoArgs.model_validate_json(call.arguments or "{}")
        except pydantic.ValidationError as error:
            return _result(needs_reply=True, error=f"Invalid arguments: {error}")
        match call.name:
            case "mute":
                self.muted = True
                return _result(
                    needs_reply=False,
                    status="muted",
                    next="Say nothing more until someone lets you talk again; keep acting if asked.",
                )
            case "unmute":
                if (refusal := self._permission.refusal()) is not None:
                    return _result(
                        needs_reply=False, error=f"{refusal} Only operators can let you talk again."
                    )
                self.muted = False
                return _result(
                    needs_reply=True,
                    status="you can talk again",
                    next="Say briefly that you are back.",
                )
            case _:
                return _result(needs_reply=True, error=f"There is no tool named {call.name}.")


def _result(*, needs_reply: bool, **fields: str) -> ToolResult:
    return ToolResult(content=json.dumps(fields), needs_reply=needs_reply)
