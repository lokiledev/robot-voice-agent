import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel


class AudioClip(RVABaseModel, arbitrary_types_allowed=True):
    samples: npt.NDArray[np.float32]
    """Mono samples in [-1, 1]."""
    sample_rate: int
    """In Hz."""

    @property
    def duration(self) -> float:
        """In seconds."""
        return len(self.samples) / self.sample_rate
