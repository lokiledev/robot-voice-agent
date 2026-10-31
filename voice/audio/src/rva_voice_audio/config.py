from pydantic import Field

from rva_pydantic import RVABaseModel


class EndpointConfig(RVABaseModel):
    """When speech starts and when the user's turn ends."""

    speech_threshold: float = Field(ge=0, le=1)
    """VAD probability above which a frame counts as speech while waiting for the user."""
    silence_threshold: float = Field(ge=0, le=1)
    """VAD probability below which a frame counts as silence while the user speaks."""
    min_speech_ms: int = Field(gt=0)
    """Speech needed to start a turn."""
    barge_in_threshold: float = Field(ge=0, le=1)
    """`speech_threshold` while the robot speaks: interrupting it needs clearer speech."""
    barge_in_min_speech_ms: int = Field(gt=0)
    """`min_speech_ms` while the robot speaks."""
    pre_roll_ms: int = Field(ge=0)
    """Audio kept from before speech was detected, so the first syllable is not cut."""
    turn_check_silence_ms: int = Field(gt=0)
    """Silence after which smart-turn is asked whether the turn is over."""
    max_silence_ms: int = Field(gt=0)
    """Silence that ends the turn whatever smart-turn says."""
    turn_complete_threshold: float = Field(ge=0, le=1)


class AudioConfig(RVABaseModel):
    input_device: str | None
    """Microphone, by name or part of it; the system default when null."""
    output_device: str | None
    """Speaker, by name or part of it; the system default when null."""
    endpoint: EndpointConfig
    echo_window_ms: int = Field(ge=0)
    """How long after the robot stops speaking a transcript of its words is dropped as an echo."""
