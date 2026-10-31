import json
from typing import Final

import pydantic
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_schemas.abc.vision import Vision
from rva_schemas.dialogue import ToolCall, ToolSpec
from rva_schemas.vision import UnknownPersonError, UnknownTrackError


class RememberPersonArgs(RVABaseModel):
    track_id: int
    """Track number of the person's face, from a robot_event or who_is_here."""
    name: str = Field(min_length=1)
    """The name they gave."""


class ForgetPersonArgs(RVABaseModel):
    name: str = Field(min_length=1)


class NoArgs(RVABaseModel):
    pass


_SPECS: Final = (
    ToolSpec(
        name="remember_person",
        description="Remember the face on a track under the name the person told you.",
        parameters=RememberPersonArgs.model_json_schema(),
    ),
    ToolSpec(
        name="who_is_here",
        description="List the people in view now, with their track numbers and names if known.",
        parameters=NoArgs.model_json_schema(),
    ),
    ToolSpec(
        name="forget_person",
        description="Forget a person's name and face.",
        parameters=ForgetPersonArgs.model_json_schema(),
    ),
)


class PeopleTools:
    """The tools the LLM uses to know who is around and remember their names."""

    specs: Final = _SPECS
    names: Final = frozenset(spec.name for spec in _SPECS)

    def __init__(self, vision: Vision) -> None:
        self._vision: Final = vision

    async def call(self, call: ToolCall) -> str:
        """Run a tool call and return its result for the LLM; its mistakes are results too."""
        try:
            match call.name:
                case "remember_person":
                    return await self._remember(
                        RememberPersonArgs.model_validate_json(call.arguments)
                    )
                case "who_is_here":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    return self._who_is_here()
                case "forget_person":
                    return await self._forget(ForgetPersonArgs.model_validate_json(call.arguments))
                case _:
                    return _result({"error": f"There is no tool named {call.name}."})
        except pydantic.ValidationError as error:
            return _result({"error": f"Invalid arguments: {error}"})

    async def _remember(self, args: RememberPersonArgs) -> str:
        try:
            remembered = await self._vision.remember(args.track_id, args.name.strip())
        except UnknownTrackError:
            return _result(
                {"error": f"No face with track {args.track_id} is in view. Call who_is_here."}
            )
        result = {"status": "remembered", "name": remembered.person.name}
        if remembered.previous is not None:
            result["previously_recognized_as"] = remembered.previous.name
        return _result(result)

    def _who_is_here(self) -> str:
        return _result(
            {
                "people": [
                    {
                        "track_id": face.track_id,
                        "name": None if face.person is None else face.person.name,
                        "nearest": face.nearest,
                    }
                    for face in self._vision.faces()
                ]
            }
        )

    async def _forget(self, args: ForgetPersonArgs) -> str:
        try:
            person = await self._vision.forget(args.name.strip())
        except UnknownPersonError:
            return _result({"error": f"Nobody named {args.name} is known."})
        return _result({"status": "forgotten", "name": person.name})


def _result(fields: pydantic.JsonValue) -> str:
    return json.dumps(fields)
