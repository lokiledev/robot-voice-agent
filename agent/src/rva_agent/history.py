from typing import Final

from rva_schemas.dialogue import Message, SystemMessage, ToolResultMessage


class History:
    """The conversation as the LLM sees it: the system prompt then the latest messages."""

    def __init__(self, system_prompt: str, max_messages: int) -> None:
        self._system: Final = SystemMessage(content=system_prompt)
        self._max_messages: Final = max_messages
        self._messages: list[Message] = []

    @property
    def messages(self) -> tuple[Message, ...]:
        return (self._system, *self._messages)

    def append(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]
            # A tool result without the call that asked for it is invalid for the LLM.
            while self._messages and isinstance(self._messages[0], ToolResultMessage):
                self._messages.pop(0)
