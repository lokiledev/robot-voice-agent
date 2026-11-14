from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from rva_pydantic import RVABaseModel
from rva_schemas.dialogue import LlmOutput, Message, ToolSpec


class LanguageModel(Protocol):
    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[LlmOutput]:
        """Generate the next assistant turn.

        Yields text as it is generated, and each tool call once its arguments are complete.
        Before tool calls that follow text, it tells that the text has ended, so that the text
        need not wait for them.
        """
        ...

    async def answer[Answer: RVABaseModel](
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], form: type[Answer]
    ) -> Answer:
        """Answer the last of `messages` as a `form`, all at once: a fact for the code to act on.

        The model can write nothing else than a `form`, so there is no reply to pick it out of.

        Args:
            messages: The conversation, ending with the question.
            tools: The conversation's tools, none of which is called: with them, a server that
                keeps what it read of the conversation has only the question left to read.
            form: What the answer is; its fields' docstrings are not shown to the model.
        """
        ...
