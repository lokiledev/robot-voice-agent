import asyncio

from rva_agent.agent import Agent
from rva_agent.config import AgentConfig
from rva_agent.tests.fakes import (
    ADMINS,
    PROFILE,
    FakeVision,
    ListRecorder,
    QueueUserInput,
    RecordingSpeech,
    ScriptedLanguageModel,
    idle_body,
    person,
)
from rva_schemas.dialogue import (
    AssistantMessage,
    NobodySeenSpeaking,
    RobotEventMessage,
    SeenSpeaking,
    TextDelta,
    ToolCall,
    UserMessage,
)
from rva_schemas.record import (
    AgentStarted,
    FaceUpdate,
    RobotLine,
    Speaker,
    TaskUpdate,
    ToolInvocation,
    UserLine,
)
from rva_schemas.vision import BoundingBox, FaceAppeared, FaceLeft, TrackedFace
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.task import TaskCancelled, TaskRunning

CONFIG = AgentConfig(
    max_history_messages=40,
    max_tool_rounds=3,
    stop_words=frozenset({"stop"}),
    mute_phrases=frozenset({"stop talking", "be quiet"}),
    unmute_phrases=frozenset({"you can talk", "talk again"}),
    regreet_after_s=300,
)
START_PICK_CUP = ToolCall(
    id="call_1", name="start_task", arguments='{"instruction": "pick up the red cup"}'
)


def _agent(
    llm: ScriptedLanguageModel,
    vla: TaskRunner,
    user: QueueUserInput,
    speech: RecordingSpeech,
    recorder: ListRecorder,
    vision: FakeVision | None,
) -> Agent:
    return Agent(
        llm=llm,
        vla=vla,
        body=idle_body(vla),
        user_input=user,
        speech=speech,
        recorder=recorder,
        vision=vision,
        profile=PROFILE,
        admins=ADMINS,
        config=CONFIG,
    )


async def test_answers_a_question() -> None:
    llm = ScriptedLanguageModel(
        [[TextDelta(text="I'm Lu"), TextDelta(text="ne. Nice to meet you.")]]
    )
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    user.say("What's your name?")
    user.leave()

    await _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        None,
    ).run()

    assert speech.said == ["I'm Lune.", "Nice to meet you."]
    started, *dialogue = recorder.entries
    assert isinstance(started, AgentStarted)
    # Without a camera, nothing about people.
    assert [tool.name for tool in started.tools] == [
        "start_task",
        "cancel_task",
        "get_task_status",
        "go_to_location",
        "stay",
        "play_animation",
    ]
    assert dialogue == [
        UserLine(text="What's your name?", speaker=None),
        RobotLine(text="I'm Lune. Nice to meet you.", interrupted=False),
    ]


async def test_keeps_talking_during_a_task_and_reports_its_end() -> None:
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Picking up the red cup."), START_PICK_CUP],
            [TextDelta(text="I'm Lune.")],
            [TextDelta(text="Done, I have the cup.")],
        ]
    )
    commands = asyncio.Queue[str]()
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm, TaskRunner(TerminalPolicy(commands, fallback=None)), user, speech, recorder, None
        ).run()
    )

    user.say("Grab the red cup please.")
    await speech.wait_for("Picking up the red cup.")
    user.say("What's your name?")
    await speech.wait_for("I'm Lune.")
    commands.put_nowait("s")
    await speech.wait_for("Done, I have the cup.")
    user.leave()
    await running

    assert speech.said == ["Picking up the red cup.", "I'm Lune.", "Done, I have the cup."]
    assert llm.requests[2][-1] == RobotEventMessage(content="You managed to pick up the red cup.")
    assert [entry for entry in recorder.entries if isinstance(entry, ToolInvocation)] == [
        ToolInvocation(
            name="start_task",
            arguments='{"instruction": "pick up the red cup"}',
            result=(
                '{"status": "started", "task": "pick up the red cup", "next": "It is running now. Its result'
                ' arrives later as a robot_event; until then do not start another task or say it is done."}'
            ),
        )
    ]


async def test_stop_word_cancels_the_task_without_the_llm() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="On it."), START_PICK_CUP]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            None,
        ).run()
    )

    user.say("Grab the red cup.")
    await speech.wait_for("On it.")
    user.say("Stop!")
    await recorder.wait_until(
        lambda entries: any(
            isinstance(entry, TaskUpdate) and entry.task.status == TaskCancelled()
            for entry in entries
        )
    )
    user.leave()
    await running

    assert speech.said == ["On it.", "Stopping."]
    assert [entry.task.status for entry in recorder.entries if isinstance(entry, TaskUpdate)] == [
        TaskRunning(),
        TaskCancelled(),
    ]
    assert len(llm.requests) == 1


async def test_barge_in_keeps_only_what_was_heard_and_skips_the_action() -> None:
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Sure. I will pick up the red cup."), START_PICK_CUP],
            [TextDelta(text="Okay.")],
        ]
    )
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=1), ListRecorder()
    user.say("Grab the red cup.")
    user.say("Actually never mind.")
    user.leave()

    await _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        None,
    ).run()

    assert llm.requests[1][-2] == AssistantMessage(content="Sure.", tool_calls=())
    assert RobotLine(text="Sure.", interrupted=True) in recorder.entries
    assert not any(isinstance(entry, ToolInvocation) for entry in recorder.entries)


BOX = BoundingBox(x=800.0, y=300.0, width=200.0, height=240.0)


async def test_asks_an_unknown_person_their_name_and_remembers_it() -> None:
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Hello! What's your name?")],
            [
                ToolCall(
                    id="call_1",
                    name="remember_person",
                    arguments='{"track_id": 8, "name": "Alice"}',
                )
            ],
            [TextDelta(text="Nice to meet you, Alice.")],
        ]
    )
    face = TrackedFace(track_id=8, box=BOX, person=None, nearest=True)
    vision = FakeVision([face])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            vision,
        ).run()
    )

    vision.show(FaceAppeared(face=face))
    await speech.wait_for("Hello!")
    user.say("I'm Alice.")
    await speech.wait_for("Nice to meet you, Alice.")
    user.leave()
    await running

    assert llm.requests[0][-1] == RobotEventMessage(
        content="An unknown person is in view (track 8, nearest). Greet them: look at them and wave."
    )
    assert (
        UserLine(text="I'm Alice.", speaker=Speaker(track_id=8, person=None, operator=False))
        in recorder.entries
    )
    assert [entry for entry in recorder.entries if isinstance(entry, ToolInvocation)] == [
        ToolInvocation(
            name="remember_person",
            arguments='{"track_id": 8, "name": "Alice"}',
            result='{"status": "remembered", "name": "Alice"}',
        )
    ]
    assert vision.faces() == (face.model_copy(update={"person": person("Alice")}),)


async def test_a_turn_is_tagged_with_the_face_seen_speaking_not_the_one_in_front() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Hi Bob.")], [TextDelta(text="Who is there?")]])
    vision = FakeVision(
        [
            TrackedFace(track_id=7, box=BOX, person=person("Alice"), nearest=True),
            TrackedFace(track_id=9, box=BOX, person=person("Bob"), nearest=False),
        ]
    )
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            vision,
        ).run()
    )

    user.say_as("Hello.", SeenSpeaking(track_id=9, person=person("Bob")))
    await speech.wait_for("Hi Bob.")
    user.say_as("Over here.", NobodySeenSpeaking())
    await speech.wait_for("Who is there?")
    user.leave()
    await running

    assert llm.requests[0][-1] == UserMessage(content="[Bob, track 9] Hello.")
    assert llm.requests[1][-1] == UserMessage(content="[someone out of view] Over here.")
    assert [entry for entry in recorder.entries if isinstance(entry, UserLine)] == [
        UserLine(text="Hello.", speaker=Speaker(track_id=9, person=person("Bob"), operator=False)),
        UserLine(text="Over here.", speaker=None),
    ]


async def test_a_departure_starts_no_turn_and_a_quick_return_is_not_greeted_again() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Hi Alice.")], [TextDelta(text="You are Alice.")]])
    alice = TrackedFace(track_id=7, box=BOX, person=person("Alice"), nearest=True)
    vision = FakeVision([])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            vision,
        ).run()
    )

    vision.show(FaceAppeared(face=alice))
    await speech.wait_for("Hi Alice.")
    vision.show(FaceLeft(track_id=7, person=person("Alice")))
    vision.show(FaceAppeared(face=alice.model_copy(update={"track_id": 9})))
    await recorder.wait_until(
        lambda entries: sum(isinstance(entry, FaceUpdate) for entry in entries) == 3
    )
    user.say("Who am I?")
    await speech.wait_for("You are Alice.")
    user.leave()
    await running

    assert speech.said == ["Hi Alice.", "You are Alice."]
    assert llm.requests[1][-3:-1] == (
        RobotEventMessage(content="Alice left (track 7)."),
        RobotEventMessage(
            content="Alice is in view (track 9, nearest, user cd736558-bc50-5e7f-a2ce-297e9d7b1465,"
            " one of your operators). Greet them: look at them and wave."
        ),
    )


async def test_muted_robot_says_nothing_until_it_may_talk_again() -> None:
    # Muting and unmuting never go through the LLM: its only turn is the question.
    llm = ScriptedLanguageModel([[TextDelta(text="It is noon.")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    user.say("Okay, now stop talking please.")
    user.say("What time is it?")
    user.say("You can talk again!")
    user.leave()

    await _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        vision=None,
    ).run()

    assert speech.said == ["Okay.", "I can talk again."]
    assert llm.requests[0][-1] == UserMessage(
        content="What time is it? (You are muted: answer with no words, only with tools if needed.)"
    )


async def test_a_failed_task_is_reported_to_the_llm_as_its_own_failure() -> None:
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Picking up the red cup."), START_PICK_CUP],
            [TextDelta(text="I couldn't pick up the red cup, my grasp slipped.")],
        ]
    )
    commands = asyncio.Queue[str]()
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(commands, fallback=None)),
            user,
            speech,
            recorder,
            vision=None,
        ).run()
    )

    user.say("Grab the red cup please.")
    await speech.wait_for("Picking up the red cup.")
    commands.put_nowait("f the grasp slipped")
    await speech.wait_for("I couldn't pick up the red cup, my grasp slipped.")
    user.leave()
    await running

    assert llm.requests[1][-1] == RobotEventMessage(
        content="You could not pick up the red cup: the grasp slipped. Say it as your own failure,"
        ' like "I couldn\'t ..."; never call it a task.'
    )
