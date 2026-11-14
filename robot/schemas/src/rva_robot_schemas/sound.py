from rva_pydantic import RVABaseModel


class SoundDirection(RVABaseModel):
    """Where the robot's microphones hear speech from, at one moment, in their own frame."""

    timestamp: float
    """When the sound was heard, in seconds of the monotonic clock the camera's frames are
    stamped with."""
    speech_azimuth: float | None
    """Direction of the speech in the array's xy plane, in radians from its x axis towards its y
    axis. `None` when it hears no speech."""
