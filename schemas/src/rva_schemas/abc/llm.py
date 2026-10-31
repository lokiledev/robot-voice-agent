from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from rva_schemas.dialogue import LlmOutput, Message, ToolSpec


class LanguageModel(Protocol):
    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[LlmOutput]:
        """Generate the next assistant turn.

        Yields text as it is generated, and each tool call once its arguments are complete.
        """
        ...
