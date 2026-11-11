from pydantic import Field

from rva_pydantic import RVABaseModel


class SpeakerAttributionConfig(RVABaseModel):
    """Which face in view said an utterance: the one seen speaking for the largest share of it."""

    history_s: float = Field(gt=0)
    """How far back who was speaking is kept: longer than the longest utterance."""
    min_share: float = Field(gt=0, le=1)
    """Share of an utterance's frames a face must be seen speaking in to have said it. Pauses
    between words count against it, so it sits well below 1."""
