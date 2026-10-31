from pydantic import TypeAdapter

from rva_schemas.dialogue import AssistantMessage, Message, ToolCall

MESSAGES = TypeAdapter(list[Message])


def test_assistant_message_with_tool_call_deserializes() -> None:
    serialized = (
        '[{"role": "assistant", "content": "On it.", "tool_calls": [{"kind": "tool_call",'
        ' "id": "call_1", "name": "start_task", "arguments": "{\\"instruction\\": \\"open the drawer\\"}"}]}]'
    )

    assert MESSAGES.validate_json(serialized) == [
        AssistantMessage(
            content="On it.",
            tool_calls=(
                ToolCall(
                    id="call_1", name="start_task", arguments='{"instruction": "open the drawer"}'
                ),
            ),
        )
    ]
