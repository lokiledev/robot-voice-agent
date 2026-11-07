from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from rva_pydantic import RVABaseModel


class ActiveSpeakerConfig(RVABaseModel):
    """Light-ASD telling which face speaks, from its lips and the microphone.

    Each frame scores the last `window_s` of each face with the sound heard meanwhile. Its two GRUs
    read the window both ways, so the last frame, which nothing follows, scores lower than the
    rest: `delay_s` scores a frame a little earlier instead, and the thresholds are below the
    model's own 0.
    """

    device: Literal["cuda", "cpu"]
    model_dir: Path
    """Where the checkpoint, converted to ONNX on first use, is kept; `~` is expanded."""
    window_s: float = Field(gt=0)
    """How much of each face and of the sound the model reads at once."""
    delay_s: float = Field(ge=0)
    """How long before the newest frame the scored one is, for the model to see what follows."""
    enter: float
    """A face starts speaking once its score is above this."""
    leave: float
    """A face stops speaking once its score is below this."""

    @model_validator(mode="after")
    def _leave_below_enter(self) -> "ActiveSpeakerConfig":
        if self.leave > self.enter:
            raise ValueError(f"leave {self.leave} is above enter {self.enter}")
        return self
