from collections.abc import AsyncGenerator, Sequence
from typing import Final, assert_never

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)
from pydantic import AnyHttpUrl, Field, JsonValue

from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import (
    AssistantMessage,
    LlmOutput,
    Message,
    RobotEventMessage,
    SystemMessage,
    TextDelta,
    TextEnd,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)


class OpenAiLanguageModelConfig(RVABaseModel):
    base_url: AnyHttpUrl
    """OpenAI-compatible endpoint, e.g. http://127.0.0.1:8080/v1 for llama-server."""
    model: str
    """Model name sent with each request; llama-server serves its one model under any name."""
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    """Longest reply, in tokens."""
    chat_template_kwargs: dict[str, JsonValue]
    """Passed to the server's chat template, e.g. {"enable_thinking": false} for Qwen."""


_EVENT_TAG: Final = "<robot_event>"


class OpenAiLanguageModel:
    """`LanguageModel` over an OpenAI-compatible chat completions server."""

    def __init__(self, config: OpenAiLanguageModelConfig) -> None:
        self._config: Final = config
        # Local servers ignore the key, but the client requires one.
        self._client: Final = AsyncOpenAI(base_url=str(config.base_url), api_key="local")

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncGenerator[LlmOutput]:
        stream = await self._client.chat.completions.create(
            model=self._config.model,
            messages=[_to_openai(message) for message in messages],
            tools=[_tool_to_openai(tool) for tool in tools],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stream=True,
            # Events are told to the model, never written by it: one it starts to write is made
            # up, and would be said aloud.
            stop=[_EVENT_TAG],
            extra_body={"chat_template_kwargs": self._config.chat_template_kwargs},
        )
        # Tool calls arrive in fragments keyed by index: id and name first, then the arguments.
        calls: dict[int, tuple[str, str, list[str]]] = {}
        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content is not None and delta.content != "":
                    yield TextDelta(text=delta.content)
                for fragment in delta.tool_calls or ():
                    if not calls:
                        # The calls take a while to write: what was said before need not wait.
                        yield TextEnd()
                    call_id, name, arguments = calls.setdefault(fragment.index, ("", "", []))
                    if fragment.function is not None:
                        name = fragment.function.name or name
                        arguments.append(fragment.function.arguments or "")
                    calls[fragment.index] = (fragment.id or call_id, name, arguments)
        for index in sorted(calls):
            call_id, name, arguments = calls[index]
            yield ToolCall(id=call_id, name=name, arguments="".join(arguments))


def _to_openai(message: Message) -> ChatCompletionMessageParam:
    match message:
        case SystemMessage():
            return {"role": "system", "content": message.content}
        case UserMessage():
            return {"role": "user", "content": message.content}
        case AssistantMessage():
            if not message.tool_calls:
                return {"role": "assistant", "content": message.content}
            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_tool_call_to_openai(call) for call in message.tool_calls],
            }
        case ToolResultMessage():
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        case RobotEventMessage():
            # Chat templates such as Qwen's only accept a system message first, so events go in a
            # user turn, tagged so the system prompt can tell the model what they are.
            return {"role": "user", "content": f"{_EVENT_TAG}{message.content}</robot_event>"}
        case _:
            assert_never(message)


def _tool_call_to_openai(call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }


def _tool_to_openai(tool: ToolSpec) -> ChatCompletionFunctionToolParam:
    parameters: dict[str, object] = dict(tool.parameters)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }
