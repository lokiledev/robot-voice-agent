from pydantic import Field

from rva_pydantic import RVABaseModel


class ReSpeakerConfig(RVABaseModel):
    """The reSpeaker XVF3800 array, asked over USB where it hears speech from."""

    frame: str
    """The array's link: the board's plane is its xy plane, with the azimuths the chip reports
    turning from its x axis towards its y axis."""
    poll_hz: float = Field(gt=0)
    """How often it is asked."""
    delay_s: float = Field(ge=0)
    """How long before it is asked the sound it describes was heard: the chip's processing."""
