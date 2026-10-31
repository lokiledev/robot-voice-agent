import asyncio
from collections.abc import AsyncGenerator

from rva_agent.history import History
from rva_agent.sentences import split_sentences
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import TaskTools
from rva_schemas.dialogue import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from rva_vla_mock.policies import TerminalPolicy
from rva_vla_mock.runner import TaskRunner


async def _chunks(*chunks: str) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk


async def test_split_sentences_across_chunks() -> None:
    chunks = _chunks("Hello th", "ere! The cup is on the", " shelf. Want it", "?\nOk")

    assert [s async for s in split_sentences(chunks)] == [
        "Hello there!",
        "The cup is on the shelf.",
        "Want it?",
        "Ok",
    ]


def test_history_drops_tool_results_orphaned_by_trimming() -> None:
    history = History(system_prompt="You are a robot.", max_messages=2)
    call = ToolCall(id="call_1", name="get_task_status", arguments="{}")

    history.append(UserMessage(content="Busy?"))
    history.append(AssistantMessage(content="", tool_calls=(call,)))
    history.append(ToolResultMessage(tool_call_id="call_1", content='{"status": "idle"}'))
    history.append(UserMessage(content="Ok."))

    assert [message.role for message in history.messages] == ["system", "user"]


async def test_tools_report_llm_mistakes_as_results() -> None:
    tools = TaskTools(
        TaskSupervisor(TaskRunner(TerminalPolicy(asyncio.Queue())), lambda task: None)
    )

    assert (await tools.call(ToolCall(id="1", name="fly", arguments="{}"))).content == (
        '{"error": "There is no tool named fly."}'
    )
    assert (
        await tools.call(ToolCall(id="2", name="start_task", arguments="{}"))
    ).content.startswith(
        '{"error": "Invalid arguments: 1 validation error for StartTaskArgs\\ninstruction\\n  Field required'
    )
    assert (await tools.call(ToolCall(id="3", name="cancel_task", arguments="{}"))).content == (
        '{"error": "No task is running."}'
    )


async def test_busy_vla_names_the_running_task() -> None:
    tools = TaskTools(
        TaskSupervisor(TaskRunner(TerminalPolicy(asyncio.Queue())), lambda task: None)
    )
    start = '{"instruction": "pick up the red cup"}'

    await tools.call(ToolCall(id="1", name="start_task", arguments=start))

    assert (await tools.call(ToolCall(id="2", name="start_task", arguments=start))).content == (
        '{"error": "Busy with \'pick up the red cup\'. Do not retry: cancel it, or wait for its robot_event."}'
    )
