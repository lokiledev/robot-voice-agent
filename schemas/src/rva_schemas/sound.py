from rva_pydantic import RVABaseModel


class HeardDirection(RVABaseModel):
    """Where a microphone array hears speech from, at one moment."""

    time: float
    """When the sound was heard, in seconds of the monotonic clock the camera's frames are stamped
    with."""
    frame: str
    """The array's link, whose frame the azimuth is in."""
    speech_azimuth: float | None
    """Direction of the speech in the array's xy plane, in radians from its x axis towards its y
    axis: a flat array tells no elevation. `None` when it hears no speech."""
