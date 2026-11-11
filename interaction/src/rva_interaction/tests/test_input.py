from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from rva_interaction.config import SpeakerAttributionConfig
from rva_interaction.input import AttributedInput
from rva_interaction.speakers import SpeakerAttribution
from rva_interaction.tests.test_speakers import frame, tracked
from rva_schemas.dialogue import SeenSpeaking, SpeechSpan, UserUtterance

NOW = datetime(2026, 11, 11, tzinfo=UTC)
SPOKEN = UserUtterance(
    text="Hello.",
    started_at=NOW,
    ended_at=NOW,
    speech=SpeechSpan(start=0.0, end=0.9),
    speaker=None,
)
TYPED = UserUtterance(text="Hi.", started_at=NOW, ended_at=NOW, speech=None, speaker=None)


class _Lines:
    async def utterances(self) -> AsyncGenerator[UserUtterance]:
        yield SPOKEN
        yield TYPED


async def test_spoken_utterances_get_their_speaker_and_typed_lines_pass() -> None:
    attribution = SpeakerAttribution(
        SpeakerAttributionConfig(history_s=30.0, min_share=0.3), score_delay_s=0.0
    )
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(4, speaking=True)))

    heard = [u async for u in AttributedInput(_Lines(), attribution).utterances()]

    assert heard == [
        SPOKEN.model_copy(update={"speaker": SeenSpeaking(track_id=4, person=None)}),
        TYPED,
    ]
