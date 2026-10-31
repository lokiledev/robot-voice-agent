import asyncio
from uuid import UUID

import pytest

from rva_agent.agent import Agent
from rva_agent.config import AgentConfig
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
from rva_schemas.profile import Operators
from rva_schemas.vision import BoundingBox, KnownPerson, TrackedFace
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.task import TaskRequest

OPERATORS = Operators(user_ids=frozenset({person("Alice").user_id}))
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
            "Alice is not one of your operators, so they cannot ask you to move.",
        ),
        (
            [_face(1, None, nearest=True)],
            "The person in front of you is not one of your operators, so they cannot ask you to move.",
        ),
        (
            [_face(1, person("Bob"), nearest=True)],
            "Bob is not one of your operators, so they cannot ask you to move.",
        ),
        (
            [_face(1, person("Bob"), nearest=True), _face(2, person("Alice"), nearest=False)],
            "Bob is not one of your operators, so they cannot ask you to move.",
        ),
        ([], "Nobody is in front of you, so nobody can ask you to move."),
    ],
)
def test_only_an_operator_in_front_may_make_the_robot_act(
    faces: list[TrackedFace], refusal: str | None
) -> None:
    assert ActionPermission(FakeVision(faces), OPERATORS).refusal() == refusal


def test_without_a_camera_anyone_may_make_the_robot_act() -> None:
    assert ActionPermission(None, OPERATORS).refusal() is None


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
            '{"error": "Bob is not one of your operators, so they cannot ask you to move.'
            ' Tell them kindly that you can talk with them but not do this."}'
        ),
    )
    assert speech.said[-1] == "Sorry, I can only do that for my operators."
    # The VLA never got the task, so it is free.
    assert (await vla.start(TaskRequest(instruction="open the drawer"))).status.state == "running"
