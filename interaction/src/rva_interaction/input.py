from collections.abc import AsyncGenerator
from typing import Final

from loguru import logger

from rva_interaction.speakers import SpeakerAttribution
from rva_schemas.abc.conversation import UserInput
from rva_schemas.dialogue import UserUtterance


class AttributedInput:
    """`UserInput` whose spoken utterances say which face in view spoke them.

    Typed lines pass as they are. Each utterance is attributed as it arrives, while what was seen
    during it is still kept, not when the agent gets to it.
    """

    def __init__(self, user_input: UserInput, attribution: SpeakerAttribution) -> None:
        self._user_input: Final = user_input
        self._attribution: Final = attribution

    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        async for utterance in self._user_input.utterances():
            if utterance.speech is None:
                yield utterance
                continue
            speaker = self._attribution.attribute(utterance.speech)
            logger.info("{!r} said by {}", utterance.text, speaker)
            yield utterance.model_copy(update={"speaker": speaker})
