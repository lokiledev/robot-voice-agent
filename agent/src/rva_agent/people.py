import json
from typing import Final

import pydantic
from pydantic import Field

from rva_agent.permission import ActionPermission
from rva_agent.tools import ToolResult
from rva_pydantic import RVABaseModel
from rva_schemas.abc.vision import Vision
from rva_schemas.dialogue import ToolCall, ToolSpec
from rva_schemas.vision import (
    NameTakenError,
    NotRecognizedError,
    UnknownPersonError,
    UnknownTrackError,
)


class RememberPersonArgs(RVABaseModel):
    track_id: int
    """Track number of the person's face, from a robot_event or who_is_here."""
    name: str = Field(min_length=1)
    """The name they gave."""


class ForgetPersonArgs(RVABaseModel):
    name: str = Field(min_length=1)


class RenamePersonArgs(RVABaseModel):
    track_id: int
    """Track number of the person's face."""
    name: str = Field(min_length=1)
    """Their name, as they confirmed it."""


class AuthorizePersonArgs(RVABaseModel):
    name: str = Field(min_length=1)
    """Name of the person to make an operator; they must be known and in view."""


class RevokePersonArgs(RVABaseModel):
    name: str = Field(min_length=1)
    """Name of the operator to revoke."""


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
        description="Forget a person's name and face entirely, when they ask you to. Never use it to correct a name.",
        parameters=ForgetPersonArgs.model_json_schema(),
    ),
    ToolSpec(
        name="rename_person",
        description="Correct the name of a person you already know, keeping who they are. The first call returns the spelling to read back; call again with the same name once they confirm it.",
        parameters=RenamePersonArgs.model_json_schema(),
    ),
    ToolSpec(
        name="list_operators",
        description="List your operators, the people who may make you act, and which of them are admins or in view.",
        parameters=NoArgs.model_json_schema(),
    ),
    ToolSpec(
        name="authorize_person",
        description="Make a known person in view one of your operators, by name, when an operator in front of you asks for it.",
        parameters=AuthorizePersonArgs.model_json_schema(),
    ),
    ToolSpec(
        name="revoke_person",
        description="Stop someone being one of your operators, when an operator in front of you asks for it.",
        parameters=RevokePersonArgs.model_json_schema(),
    ),
)


class PeopleTools:
    """The tools the LLM uses to know who is around and remember their names."""

    specs: Final = _SPECS
    names: Final = frozenset(spec.name for spec in _SPECS)

    def __init__(self, vision: Vision, permission: ActionPermission) -> None:
        self._vision: Final = vision
        self._permission: Final = permission
        # A rename waiting for the person to confirm the spelling: (track id, new name).
        self._pending_rename: tuple[int, str] | None = None

    async def call(self, call: ToolCall) -> ToolResult:
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
                case "rename_person":
                    return await self._rename(RenamePersonArgs.model_validate_json(call.arguments))
                case "list_operators":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    return await self._list_operators()
                case "authorize_person":
                    return await self._authorize(
                        AuthorizePersonArgs.model_validate_json(call.arguments)
                    )
                case "revoke_person":
                    return await self._revoke(RevokePersonArgs.model_validate_json(call.arguments))
                case _:
                    return _result({"error": f"There is no tool named {call.name}."})
        except pydantic.ValidationError as error:
            return _result({"error": f"Invalid arguments: {error}"})

    async def _remember(self, args: RememberPersonArgs) -> ToolResult:
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

    def _who_is_here(self) -> ToolResult:
        return _result(
            {
                "people": [
                    {
                        "track_id": face.track_id,
                        "name": None if face.person is None else face.person.name,
                        "user_id": None if face.person is None else str(face.person.user_id),
                        "nearest": face.nearest,
                    }
                    for face in self._vision.faces()
                ]
            }
        )

    async def _forget(self, args: ForgetPersonArgs) -> ToolResult:
        try:
            person = await self._vision.forget(args.name.strip())
        except UnknownPersonError:
            return _result({"error": f"Nobody named {args.name} is known."})
        return _result({"status": "forgotten", "name": person.name})

    async def _rename(self, args: RenamePersonArgs) -> ToolResult:
        name = args.name.strip()
        # Names are easy to mishear, so a rename only happens once the person heard the spelling.
        if self._pending_rename != (args.track_id, name):
            self._pending_rename = (args.track_id, name)
            return _result(
                {
                    "status": "needs confirmation",
                    "spelling": " ".join(name.upper()),
                    "next": "Spell it to them letter by letter and ask if it is right. If it is, call rename_person again with the same name; if not, ask them to spell it.",
                }
            )
        self._pending_rename = None
        try:
            person = await self._vision.rename(args.track_id, name)
        except UnknownTrackError:
            return _result(
                {"error": f"No face with track {args.track_id} is in view. Call who_is_here."}
            )
        except NotRecognizedError:
            return _result({"error": "You do not know this person yet: use remember_person."})
        except NameTakenError:
            return _result({"error": f"Someone else is already called {args.name}."})
        return _result({"status": "renamed", "name": person.name})

    async def _list_operators(self) -> ToolResult:
        names = {person.user_id: person.name for person in await self._vision.people()}
        in_view = {face.person.user_id for face in self._vision.faces() if face.person is not None}
        operators: list[pydantic.JsonValue] = [
            {
                "name": names.get(user_id),
                "admin": self._permission.is_admin(user_id),
                "in_view": user_id in in_view,
            }
            for user_id in sorted(self._permission.operators())
        ]
        return _result({"operators": operators})

    async def _authorize(self, args: AuthorizePersonArgs) -> ToolResult:
        requester = self._permission.operator_in_front()
        if isinstance(requester, str):
            return _result({"error": f"{requester} Only operators can authorize people."})
        wanted = args.name.strip().casefold()
        person = next(
            (
                face.person
                for face in self._vision.faces()
                if face.person is not None and face.person.name.casefold() == wanted
            ),
            None,
        )
        if person is None:
            return _result(
                {
                    "error": f"Nobody named {args.name} is in view. They must be in view; if you do"
                    " not know them yet, ask their name and call remember_person first."
                }
            )
        if self._permission.is_operator(person.user_id):
            return _result({"status": "already an operator", "name": person.name})
        await self._vision.grant_operator(person.user_id, requester.user_id)
        return _result({"status": "authorized", "name": person.name})

    async def _revoke(self, args: RevokePersonArgs) -> ToolResult:
        requester = self._permission.operator_in_front()
        if isinstance(requester, str):
            return _result({"error": f"{requester} Only operators can revoke people."})
        wanted = args.name.strip().casefold()
        person = next((p for p in await self._vision.people() if p.name.casefold() == wanted), None)
        if person is None:
            return _result({"error": f"Nobody named {args.name} is known."})
        if self._permission.is_admin(person.user_id):
            return _result(
                {
                    "error": f"{person.name} is an admin: admins are only changed in the admins file, not by voice."
                }
            )
        if not await self._vision.revoke_operator(person.user_id):
            return _result({"error": f"{person.name} is not one of your operators."})
        return _result({"status": "revoked", "name": person.name})


def _result(fields: pydantic.JsonValue) -> ToolResult:
    return ToolResult(content=json.dumps(fields), needs_reply=True)
