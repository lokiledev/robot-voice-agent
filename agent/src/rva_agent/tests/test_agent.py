import asyncio
from collections.abc import AsyncGenerator, Sequence

from rva_agent.agent import Agent
from rva_agent.config import AgentConfig
from rva_agent.tests.fakes import (
    ADMINS,
    PROFILE,
    FakeEngagement,
    FakeVision,
    ListRecorder,
    QueueUserInput,
    RecordingSpeech,
    ScriptedLanguageModel,
    idle_body,
    person,
)
from rva_schemas.abc.llm import LanguageModel
from rva_schemas.dialogue import (
    AssistantMessage,
    LlmOutput,
    Message,
    NobodySeenSpeaking,
    RobotEventMessage,
    SeenSpeaking,
    TextDelta,
    TextEnd,
    ToolCall,
    ToolSpec,
    UserMessage,
)
from rva_schemas.interaction import Disengaged, Engaged
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
    llm: LanguageModel,
    vla: TaskRunner,
    user: QueueUserInput,
    speech: RecordingSpeech,
    recorder: ListRecorder,
    vision: FakeVision | None,
    engagement: FakeEngagement | None,
) -> Agent:
    return Agent(
        llm=llm,
        vla=vla,
        body=idle_body(vla),
        user_input=user,
        speech=speech,
        recorder=recorder,
        vision=vision,
        engagement=engagement,
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
            llm,
            TaskRunner(TerminalPolicy(commands, fallback=None)),
            user,
            speech,
            recorder,
            None,
            None,
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
    vision, engagement = FakeVision([face]), FakeEngagement([])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = asyncio.create_task(
        _agent(
            llm,
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            vision,
            engagement,
        ).run()
    )

    engagement.show(Engaged(track_id=8, person=None))
    await speech.wait_for("Hello!")
    user.say("I'm Alice.")
    await speech.wait_for("Nice to meet you, Alice.")
    user.leave()
    await running

    assert llm.requests[0][-1] == RobotEventMessage(
        content="An unknown person (track 8) came to you and is looking at you. Greet them: look"
        " at them and wave."
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
    llm = ScriptedLanguageModel([[TextDelta(text="Hi Bob.")]])
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
            None,
        ).run()
    )

    user.say_as("Hello.", SeenSpeaking(track_id=9, person=person("Bob"), to_robot=True))
    await speech.wait_for("Hi Bob.")
    user.leave()
    await running

    assert llm.requests[0][-1] == UserMessage(content="[Bob, track 9] Hello.")
    assert [entry for entry in recorder.entries if isinstance(entry, UserLine)] == [
        UserLine(text="Hello.", speaker=Speaker(track_id=9, person=person("Bob"), operator=False)),
    ]


def _running(
    llm: ScriptedLanguageModel,
    user: QueueUserInput,
    speech: RecordingSpeech,
    recorder: ListRecorder,
    vision: FakeVision,
    engagement: FakeEngagement,
) -> asyncio.Task[None]:
    vla = TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None))
    return asyncio.create_task(_agent(llm, vla, user, speech, recorder, vision, engagement).run())


ALICE_ID = "cd736558-bc50-5e7f-a2ce-297e9d7b1465"


async def test_someone_passing_by_is_recorded_but_starts_no_turn() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Nobody came to me.")]])
    alice = TrackedFace(track_id=7, box=BOX, person=person("Alice"), nearest=True)
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    vision = FakeVision([])
    running = _running(llm, user, speech, recorder, vision, FakeEngagement([]))

    vision.show(FaceAppeared(face=alice))
    vision.show(FaceLeft(track_id=7, person=person("Alice")))
    await recorder.wait_until(
        lambda entries: sum(isinstance(entry, FaceUpdate) for entry in entries) == 2
    )
    user.say("Who came?")
    await speech.wait_for("Nobody came to me.")
    user.leave()
    await running

    assert [message.role for message in llm.requests[0]] == ["system", "user"]


async def test_a_quick_return_is_not_greeted_again() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Hi Alice.")], [TextDelta(text="You are Alice.")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    engagement = FakeEngagement([])
    running = _running(llm, user, speech, recorder, FakeVision([]), engagement)

    engagement.show(Engaged(track_id=7, person=person("Alice")))
    await speech.wait_for("Hi Alice.")
    engagement.show(Disengaged(track_id=7, person=person("Alice"), left=True))
    engagement.show(Engaged(track_id=9, person=person("Alice")))
    user.say("Who am I?")
    await speech.wait_for("You are Alice.")
    user.leave()
    await running

    assert speech.said == ["Hi Alice.", "You are Alice."]
    assert llm.requests[0][-1] == RobotEventMessage(
        content=f"Alice (track 7, user {ALICE_ID}, one of your operators) came to you and is"
        " looking at you. Greet them: look at them and wave."
    )
    assert llm.requests[1][-3:-1] == (
        RobotEventMessage(content="Alice (track 7) left."),
        RobotEventMessage(
            content=f"Alice (track 9, user {ALICE_ID}, one of your operators) is paying attention"
            " to you again."
        ),
    )


async def test_what_is_not_said_to_the_robot_is_not_heard() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="It is noon.")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    running = _running(llm, user, speech, recorder, FakeVision([]), FakeEngagement([]))

    user.say_as("Shall we go?", SeenSpeaking(track_id=9, person=person("Bob"), to_robot=False))
    user.say_as("Hey, robot!", NobodySeenSpeaking())
    user.say_as("What time is it?", SeenSpeaking(track_id=7, person=None, to_robot=True))
    await speech.wait_for("It is noon.")
    user.leave()
    await running

    assert llm.requests[0][1:] == (
        UserMessage(content="[unknown person, track 7] What time is it?"),
    )
    assert [entry.text for entry in recorder.entries if isinstance(entry, UserLine)] == [
        "What time is it?"
    ]


async def test_a_reply_stops_when_its_listener_turns_away_and_is_picked_up_when_they_are_back() -> (
    None
):
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="There are three. The first is the lobby. The second is the bar.")],
            [TextDelta(text="As I was saying, the second is the bar.")],
        ]
    )
    user, recorder = QueueUserInput(), ListRecorder()
    engagement = FakeEngagement([7])

    speech = RecordingSpeech(interrupt_after=None)

    def turn_away(sentence: str) -> None:
        if sentence == "There are three.":
            engagement.show(Disengaged(track_id=7, person=person("Alice"), left=False))

    speech.after_sentence = turn_away
    running = _running(llm, user, speech, recorder, FakeVision([]), engagement)

    user.say_as("Which rooms?", SeenSpeaking(track_id=7, person=person("Alice"), to_robot=True))
    await speech.wait_for("There are three.")
    engagement.show(Engaged(track_id=7, person=person("Alice")))
    await speech.wait_for("As I was saying, the second is the bar.")
    user.leave()
    await running

    assert speech.said == ["There are three.", "As I was saying, the second is the bar."]
    assert RobotLine(text="There are three.", interrupted=True) in recorder.entries
    assert llm.requests[1][-4:] == (
        AssistantMessage(content="There are three.", tool_calls=()),
        RobotEventMessage(content="You stopped talking there: they were no longer listening."),
        RobotEventMessage(content="Alice (track 7) is no longer paying attention to you."),
        RobotEventMessage(
            content=f"Alice (track 7, user {ALICE_ID}, one of your operators) is listening to you"
            " again. You had stopped in the middle of your answer: pick it up if it still"
            " matters, briefly."
        ),
    )


async def test_a_reply_cut_short_is_picked_up_when_its_listener_is_back_on_another_track() -> None:
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Once upon a time. There was a robot.")],
            [TextDelta(text="So, there was a robot.")],
        ]
    )
    user, recorder = QueueUserInput(), ListRecorder()
    engagement = FakeEngagement([7])
    speech = RecordingSpeech(interrupt_after=None)

    def walk_away(sentence: str) -> None:
        if sentence == "Once upon a time.":
            engagement.show(Disengaged(track_id=7, person=person("Alice"), left=True))

    speech.after_sentence = walk_away
    running = _running(llm, user, speech, recorder, FakeVision([]), engagement)

    user.say_as("A story?", SeenSpeaking(track_id=7, person=person("Alice"), to_robot=True))
    await speech.wait_for("Once upon a time.")
    # Out of view and back, Alice is on a new track.
    engagement.show(Engaged(track_id=8, person=person("Alice")))
    await speech.wait_for("So, there was a robot.")
    user.leave()
    await running

    assert speech.said == ["Once upon a time.", "So, there was a robot."]
    assert llm.requests[1][-1] == RobotEventMessage(
        content=f"Alice (track 8, user {ALICE_ID}, one of your operators) is listening to you"
        " again. You had stopped in the middle of your answer: pick it up if it still matters,"
        " briefly."
    )


async def test_a_stop_word_cuts_off_the_reply_being_said() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Hello.")], [TextDelta(text="Okay.")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    user.say("Hello.")
    user.say("Stop!")
    user.leave()

    await _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        None,
        None,
    ).run()

    assert speech.interruptions == 1


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
        engagement=None,
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
            engagement=None,
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


async def test_a_tool_called_without_a_word_ends_the_turn() -> None:
    # Asked again with nothing left to say, a small model makes up the event it expects next.
    llm = ScriptedLanguageModel([[ToolCall(id="call_1", name="stay", arguments="{}")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    user.say("Okay.")
    user.leave()

    await _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        None,
        None,
    ).run()

    assert speech.said == []
    assert len(llm.requests) == 1


async def test_a_call_that_failed_is_not_run_again_in_the_same_turn() -> None:
    again = ToolCall(id="call_2", name="start_task", arguments='{"instruction": "wipe the table"}')
    llm = ScriptedLanguageModel(
        [
            [TextDelta(text="Picking up the red cup."), START_PICK_CUP],
            [TextDelta(text="Wiping the table."), again],
            # Told it is busy, it says it again and retries, as small models do.
            [TextDelta(text="Now I wipe the table."), again.model_copy(update={"id": "call_3"})],
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
            None,
            None,
        ).run()
    )

    user.say("Grab the red cup please.")
    await speech.wait_for("Picking up the red cup.")
    user.say("And wipe the table.")
    await speech.wait_for("Now I wipe the table.")
    user.say("stop")
    user.leave()
    await running

    assert [entry.result for entry in recorder.entries if isinstance(entry, ToolInvocation)][
        1:
    ] == [
        '{"error": "Busy with \'pick up the red cup\'. Do not retry: cancel it, or wait for its'
        ' robot_event."}',
        '{"error": "You just called this, and it was not run again. Do not call it again now."}',
    ]
    # The turn ended there: the LLM was not asked a fourth time.
    assert len(llm.requests) == 3


async def test_what_is_said_does_not_wait_for_the_tool_calls_that_follow_it() -> None:
    speech = RecordingSpeech(interrupt_after=None)
    heard = asyncio.Event()
    speech.after_sentence = lambda sentence: heard.set()

    class SlowToWriteCalls:
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec]
        ) -> AsyncGenerator[LlmOutput]:
            yield TextDelta(text="Picking up the red cup.")
            yield TextEnd()
            # The call is only written once the sentence is heard.
            await heard.wait()
            yield START_PICK_CUP

    user, recorder = QueueUserInput(), ListRecorder()
    user.say("Grab the red cup please.")
    user.leave()

    await asyncio.wait_for(
        _agent(
            SlowToWriteCalls(),
            TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
            user,
            speech,
            recorder,
            None,
            None,
        ).run(),
        timeout=5,
    )

    assert speech.said == ["Picking up the red cup."]
    assert RobotLine(text="Picking up the red cup.", interrupted=False) in recorder.entries
    assert [e.name for e in recorder.entries if isinstance(e, ToolInvocation)] == ["start_task"]


async def test_warming_up_has_the_llm_read_the_prompt_and_says_nothing() -> None:
    llm = ScriptedLanguageModel([[TextDelta(text="Hello."), TextDelta(text=" I am Lune.")]])
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    agent = _agent(
        llm,
        TaskRunner(TerminalPolicy(asyncio.Queue(), fallback=None)),
        user,
        speech,
        recorder,
        None,
        None,
    )

    await agent.warm_up()

    (request,) = llm.requests
    assert [message.role for message in request] == ["system", "user"]
    assert speech.said == []
    assert recorder.entries == []
