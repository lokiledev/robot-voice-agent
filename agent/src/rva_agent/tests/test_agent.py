import asyncio

from rva_agent.agent import Agent
from rva_agent.config import AgentConfig
from rva_agent.tests.fakes import (
    ListRecorder,
    QueueUserInput,
    RecordingSpeech,
    ScriptedLanguageModel,
)
from rva_schemas.dialogue import AssistantMessage, RobotEventMessage, TextDelta, ToolCall
from rva_schemas.profile import Capabilities, Character, Environment, Profile, Skill
from rva_schemas.record import DialogueLine, TaskUpdate, ToolInvocation
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_schemas.task import TaskCancelled

PROFILE = Profile(
    character=Character(
        name="Lune", personality="Calm.", speaking_style="Short.", verbosity="terse"
    ),
    environment=Environment(location="Lab", description="A lab.", places=(), objects=(), people=()),
    capabilities=Capabilities(
        skills=(
            Skill(name="pick", description="Pick up an object.", examples=("pick up the red cup",)),
        ),
        constraints=(),
    ),
)
CONFIG = AgentConfig(max_history_messages=40, max_tool_rounds=3, stop_words=frozenset({"stop"}))
START_PICK_CUP = ToolCall(
    id="call_1", name="start_task", arguments='{"instruction": "pick up the red cup"}'
)


def _agent(
    llm: ScriptedLanguageModel,
    vla: TaskRunner,
    user: QueueUserInput,
    speech: RecordingSpeech,
    recorder: ListRecorder,
) -> Agent:
    return Agent(
        llm=llm,
        vla=vla,
        user_input=user,
        speech=speech,
        recorder=recorder,
        profile=PROFILE,
        config=CONFIG,
    )


async def test_answers_a_question() -> None:
    llm = ScriptedLanguageModel(
        [[TextDelta(text="I'm Lu"), TextDelta(text="ne. Nice to meet you.")]]
    )
    user, speech, recorder = QueueUserInput(), RecordingSpeech(interrupt_after=None), ListRecorder()
    user.say("What's your name?")
    user.leave()

    await _agent(llm, TaskRunner(TerminalPolicy(asyncio.Queue())), user, speech, recorder).run()

    assert speech.said == ["I'm Lune.", "Nice to meet you."]
    assert recorder.entries == [
        DialogueLine(speaker="user", text="What's your name?"),
        DialogueLine(speaker="robot", text="I'm Lune. Nice to meet you."),
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
        _agent(llm, TaskRunner(TerminalPolicy(commands)), user, speech, recorder).run()
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
    assert llm.requests[2][-1] == RobotEventMessage(
        content="Your body finished: pick up the red cup. It worked."
    )
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
        _agent(llm, TaskRunner(TerminalPolicy(asyncio.Queue())), user, speech, recorder).run()
    )

    user.say("Grab the red cup.")
    await speech.wait_for("On it.")
    user.say("Stop!")
    await recorder.wait_until(
        lambda entries: any(isinstance(entry, TaskUpdate) for entry in entries)
    )
    user.leave()
    await running

    assert speech.said == ["On it.", "Stopping."]
    assert [entry.task.status for entry in recorder.entries if isinstance(entry, TaskUpdate)] == [
        TaskCancelled()
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

    await _agent(llm, TaskRunner(TerminalPolicy(asyncio.Queue())), user, speech, recorder).run()

    assert llm.requests[1][-2] == AssistantMessage(content="Sure.", tool_calls=())
    assert not any(isinstance(entry, ToolInvocation) for entry in recorder.entries)
