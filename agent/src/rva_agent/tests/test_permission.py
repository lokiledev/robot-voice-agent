import asyncio
import json
from uuid import UUID

import pytest

from rva_agent.agent import Agent
from rva_agent.config import AgentConfig
from rva_agent.people import PeopleTools
from rva_agent.permission import ActionPermission
from rva_agent.tests.fakes import (
    PROFILE,
    FakeVision,
    ListRecorder,
    QueueUserInput,
    RecordingSpeech,
    ScriptedLanguageModel,
    person,
)
from rva_schemas.dialogue import TextDelta, ToolCall, ToolResultMessage
from rva_schemas.profile import Admins
from rva_schemas.vision import BoundingBox, KnownPerson, TrackedFace
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.task import TaskRequest

ADMINS = Admins(user_ids=frozenset({person("Alice").user_id}))
BOX = BoundingBox(x=0, y=0, width=100, height=100)


def _face(track_id: int, who: KnownPerson | None, nearest: bool) -> TrackedFace:
    return TrackedFace(track_id=track_id, box=BOX, person=who, nearest=nearest)


@pytest.mark.parametrize(
    ("faces", "refusal"),
    [
        ([_face(1, person("Alice"), nearest=True)], None),
        # Someone else with the same name is a different user.
        (
            [_face(1, KnownPerson(user_id=UUID(int=1), name="Alice"), nearest=True)],
            "Alice is not one of your operators.",
        ),
        (
            [_face(1, None, nearest=True)],
            "The person in front of you is not one of your operators.",
        ),
        (
            [_face(1, person("Bob"), nearest=True)],
            "Bob is not one of your operators.",
        ),
        (
            [_face(1, person("Bob"), nearest=True), _face(2, person("Alice"), nearest=False)],
            "Bob is not one of your operators.",
        ),
        ([], "Nobody is in front of you."),
    ],
)
def test_only_an_operator_in_front_may_make_the_robot_act(
    faces: list[TrackedFace], refusal: str | None
) -> None:
    assert ActionPermission(FakeVision(faces), ADMINS).refusal() == refusal


def test_without_a_camera_anyone_may_make_the_robot_act() -> None:
    assert ActionPermission(None, ADMINS).refusal() is None


async def test_a_stranger_cannot_start_a_task_but_can_still_talk() -> None:
    start = ToolCall(
        id="call_1", name="start_task", arguments='{"instruction": "pick up the red cup"}'
    )
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Picking up the red cup."), start],
            [TextDelta(text="Sorry, I can only do that for my operators.")],
        ]
    )
    vla = TaskRunner(TerminalPolicy(asyncio.Queue()))
    user, speech = QueueUserInput(), RecordingSpeech(interrupt_after=None)
    user.say("Grab the red cup.")
    user.leave()

    await Agent(
        llm=llm,
        vla=vla,
        user_input=user,
        speech=speech,
        recorder=ListRecorder(),
        vision=FakeVision([_face(4, person("Bob"), nearest=True)]),
        profile=PROFILE,
        config=AgentConfig(
            max_history_messages=40,
            max_tool_rounds=3,
            stop_words=frozenset({"stop"}),
            regreet_after_s=60,
        ),
    ).run()

    assert llm.requests[1][-1] == ToolResultMessage(
        tool_call_id="call_1",
        content=(
            '{"error": "Bob is not one of your operators. Only operators can ask you to move.'
            ' Tell them kindly that you can talk with them but not do this."}'
        ),
    )
    assert speech.said[-1] == "Sorry, I can only do that for my operators."
    # The VLA never got the task, so it is free.
    assert (await vla.start(TaskRequest(instruction="open the drawer"))).status.state == "running"


def _people_tools(faces: list[TrackedFace]) -> tuple[FakeVision, PeopleTools]:
    vision = FakeVision(faces)
    return vision, PeopleTools(vision, ActionPermission(vision, ADMINS))


async def _call(tools: PeopleTools, name: str, arguments: str) -> str:
    return (await tools.call(ToolCall(id="1", name=name, arguments=arguments))).content


async def test_an_operator_in_front_authorizes_a_known_person_in_view() -> None:
    bob = person("Bob")
    vision, tools = _people_tools(
        [_face(1, person("Alice"), nearest=True), _face(2, bob, nearest=False)]
    )

    assert await _call(tools, "authorize_person", '{"name": "bob"}') == (
        '{"status": "authorized", "name": "Bob"}'
    )
    assert vision.granted == {bob.user_id: person("Alice").user_id}
    assert await _call(tools, "authorize_person", '{"name": "Bob"}') == (
        '{"status": "already an operator", "name": "Bob"}'
    )


@pytest.mark.parametrize(
    ("faces", "error"),
    [
        (
            [_face(1, person("Bob"), nearest=True), _face(2, person("Carol"), nearest=False)],
            "Bob is not one of your operators. Only operators can authorize people.",
        ),
        (
            [_face(1, person("Alice"), nearest=True), _face(2, None, nearest=False)],
            "Nobody named Carol is in view. They must be in view; if you do not know them yet,"
            " ask their name and call remember_person first.",
        ),
    ],
)
async def test_authorizing_needs_an_operator_in_front_and_a_known_face(
    faces: list[TrackedFace], error: str
) -> None:
    vision, tools = _people_tools(faces)

    assert await _call(tools, "authorize_person", '{"name": "Carol"}') == f'{{"error": "{error}"}}'
    assert vision.granted == {}


async def test_an_operator_revokes_a_granted_operator_but_not_an_admin() -> None:
    bob = person("Bob")
    vision, tools = _people_tools(
        [_face(1, person("Alice"), nearest=True), _face(2, bob, nearest=False)]
    )
    await vision.grant_operator(bob.user_id, person("Alice").user_id)

    assert await _call(tools, "revoke_person", '{"name": "Alice"}') == (
        '{"error": "Alice is an admin: admins can only be changed in your profile."}'
    )
    assert await _call(tools, "revoke_person", '{"name": "bob"}') == (
        '{"status": "revoked", "name": "Bob"}'
    )
    assert await _call(tools, "revoke_person", '{"name": "Bob"}') == (
        '{"error": "Bob is not one of your operators."}'
    )


async def test_list_operators_names_admins_and_granted_operators() -> None:
    bob = person("Bob")
    vision, tools = _people_tools([_face(1, person("Alice"), nearest=True)])
    vision.known["Bob"] = bob
    await vision.grant_operator(bob.user_id, person("Alice").user_id)

    listed = json.loads(await _call(tools, "list_operators", "{}"))["operators"]

    assert sorted(listed, key=lambda operator: operator["name"]) == [
        {"name": "Alice", "admin": True, "in_view": True},
        {"name": "Bob", "admin": False, "in_view": False},
    ]


async def test_renaming_keeps_the_user_and_their_operator_status() -> None:
    alice = person("Alice")
    vision, tools = _people_tools([_face(1, alice, nearest=True)])

    rename = '{"track_id": 1, "name": "Alicia"}'

    assert json.loads(await _call(tools, "rename_person", rename))["spelling"] == "A L I C I A"
    assert vision.faces()[0].person == alice
    assert await _call(tools, "rename_person", rename) == '{"status": "renamed", "name": "Alicia"}'

    (face,) = vision.faces()
    assert face.person == KnownPerson(user_id=alice.user_id, name="Alicia")
    assert ActionPermission(vision, ADMINS).refusal() is None
