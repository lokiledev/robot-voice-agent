from collections.abc import AsyncGenerator
from typing import Final

from rva_interaction.interaction import Interaction
from rva_schemas.abc.conversation import UserInput
from rva_schemas.dialogue import UserUtterance


class AttributedInput:
    """`UserInput` whose spoken utterances say which face spoke them, and whether to the robot.

    Each utterance is judged as it arrives, while what was seen during it is still kept, not
    when the agent gets to it.
    """

    def __init__(self, user_input: UserInput, interaction: Interaction) -> None:
        self._user_input: Final = user_input
        self._interaction: Final = interaction

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        async for utterance in self._user_input.utterances():
            yield self._interaction.hear(utterance)
