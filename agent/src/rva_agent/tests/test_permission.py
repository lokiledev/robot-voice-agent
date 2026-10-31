import asyncio
import json
from uuid import UUID

import pytest

from rva_agent.agent import Agent
from rva_agent.body import BodySupervisor, BodyTools
from rva_agent.config import AgentConfig
from rva_agent.people import PeopleTools
from rva_agent.permission import ActionPermission
from rva_agent.tasks import TaskSupervisor
from rva_agent.tests.fakes import (
    PROFILE,
    FakeVision,
    ListRecorder,
    QueueUserInput,
    RecordingSpeech,
    ScriptedLanguageModel,
    idle_body,
    person,
)
from rva_schemas.dialogue import TextDelta, ToolCall, ToolResultMessage
from rva_schemas.profile import Admins
from rva_schemas.vision import BoundingBox, KnownPerson, TrackedFace
from rva_vla_mock.body import BodyMock
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
    vla = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    user, speech = QueueUserInput(), RecordingSpeech(interrupt_after=None)
    user.say("Grab the red cup.")
    user.leave()

    await Agent(
        llm=llm,
        vla=vla,
        body=idle_body(vla),
        user_input=user,
        speech=speech,
        recorder=ListRecorder(),
        vision=FakeVision([_face(4, person("Bob"), nearest=True)]),
        profile=PROFILE,
        config=AgentConfig(
            max_history_messages=40,
            max_tool_rounds=3,
            stop_words=frozenset({"stop"}),
            mute_phrases=frozenset({"stop talking", "be quiet"}),
            unmute_phrases=frozenset({"you can talk", "talk again"}),
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


def _body_tools(
    faces: list[TrackedFace], vla: TaskRunner, commands: asyncio.Queue[str]
) -> tuple[BodyTools, TaskSupervisor]:
    policy = TerminalPolicy(commands, fallback=None)
    tasks = TaskSupervisor(vla, lambda task: None)
    vision = FakeVision(faces)
    body = BodySupervisor(BodyMock(vla, policy, policy), lambda motion: None)
    tools = BodyTools(
        body,
        tasks,
        ActionPermission(vision, ADMINS),
        vision,
        ("kitchen counter", "shelf"),
        ("wave_right_hand", "dance"),
    )
    return tools, tasks


async def _body_call(tools: BodyTools, name: str, arguments: str) -> str:
    return (await tools.call(ToolCall(id="1", name=name, arguments=arguments))).content


async def test_anyone_may_ask_for_a_gesture_but_only_an_operator_for_a_walk() -> None:
    tools, _ = _body_tools(
        [_face(1, person("Bob"), nearest=True)],
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        asyncio.Queue(),
    )

    assert await _body_call(tools, "go_to_location", '{"location": "shelf"}') == (
        '{"error": "Bob is not one of your operators. Only operators can ask you to walk somewhere."}'
    )
    assert await _body_call(tools, "play_animation", '{"animation": "wave_right_hand"}') == (
        '{"status": "started", "motion": "play the wave_right_hand animation",'
        ' "next": "You are told when it ends."}'
    )


async def test_tasks_walks_and_animations_share_the_body() -> None:
    commands = asyncio.Queue[str]()
    vla = TaskRunner(TerminalPolicy(commands, fallback=None))
    tools, tasks = _body_tools([_face(1, person("Alice"), nearest=True)], vla, commands)

    assert await _body_call(tools, "go_to_location", '{"location": "Kitchen Counter"}') == (
        '{"status": "started", "motion": "go to the kitchen counter",'
        ' "next": "You are told when it ends."}'
    )
    assert await _body_call(tools, "play_animation", '{"animation": "dance"}') == (
        '{"error": "Your body is busy: go to the kitchen counter. Call stay first or wait."}'
    )
    await _body_call(tools, "stay", "{}")
    await tasks.start("pick up the red cup")
    assert await _body_call(tools, "play_animation", '{"animation": "dance"}') == (
        '{"error": "Your body is busy with \'pick up the red cup\'. Cancel it first or wait."}'
    )


async def test_unknown_places_and_animations_are_listed_back() -> None:
    tools, _ = _body_tools(
        [_face(1, person("Alice"), nearest=True)],
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        asyncio.Queue(),
    )

    assert await _body_call(tools, "go_to_location", '{"location": "garden"}') == (
        '{"error": "garden is not a place you know. Places: kitchen counter, shelf."}'
    )
    assert await _body_call(tools, "play_animation", '{"animation": "backflip"}') == (
        '{"error": "There is no animation backflip. Animations: wave_right_hand, dance."}'
    )
    assert (
        await _body_call(tools, "look_at_user", f'{{"user_id": "{person("Alice").user_id}"}}')
        == '{"status": "looking at Alice"}'
    )


async def test_each_user_turn_names_who_was_in_front() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Sure.")], [TextDelta(text="Hi.")]])
    vision = FakeVision([_face(1, person("Bob"), nearest=True), _face(2, None, nearest=False)])
    vla = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    user = QueueUserInput()
    agent = Agent(
        llm=llm,
        vla=vla,
        body=idle_body(vla),
        user_input=user,
        speech=RecordingSpeech(interrupt_after=None),
        recorder=ListRecorder(),
        vision=vision,
        profile=PROFILE,
        config=AgentConfig(
            max_history_messages=40,
            max_tool_rounds=3,
            stop_words=frozenset({"stop"}),
            mute_phrases=frozenset({"stop talking", "be quiet"}),
            unmute_phrases=frozenset({"you can talk", "talk again"}),
            regreet_after_s=60,
        ),
    )
    running = asyncio.create_task(agent.run())
    user.say("Go to the shelf.")
    await asyncio.sleep(0.05)
    vision.set_faces([_face(1, person("Bob"), nearest=False), _face(2, None, nearest=True)])
    user.say("Hello?")
    user.leave()
    await running

    assert [request[-1].content for request in llm.requests] == [
        "[Bob, track 1] Go to the shelf.",
        "[unknown person, track 2] Hello?",
    ]


async def test_anyone_may_mute_but_only_an_operator_unmutes() -> None:
    vision = FakeVision([_face(1, person("Bob"), nearest=True)])
    vla = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    user, speech = QueueUserInput(), RecordingSpeech(interrupt_after=None)
    running = asyncio.create_task(
        Agent(
            llm=ScriptedLanguageModel([]),
            vla=vla,
            body=idle_body(vla),
            user_input=user,
            speech=speech,
            recorder=ListRecorder(),
            vision=vision,
            profile=PROFILE,
            config=AgentConfig(
                max_history_messages=40,
                max_tool_rounds=3,
                stop_words=frozenset({"stop"}),
                mute_phrases=frozenset({"stop talking"}),
                unmute_phrases=frozenset({"talk again"}),
                regreet_after_s=60,
            ),
        ).run()
    )

    user.say("Stop talking.")
    await speech.wait_for("Okay.")
    user.say("You can talk again.")
    await asyncio.sleep(0.05)
    assert speech.said == ["Okay."]

    vision.set_faces([_face(2, person("Alice"), nearest=True)])
    user.say("You can talk again.")
    user.leave()
    await running

    assert speech.said == ["Okay.", "I can talk again."]
