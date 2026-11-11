from pathlib import Path

from rva_pydantic import RVABaseModel


class ReachyMiniDescriptionConfig(RVABaseModel):
    """Pollen Robotics' MJCF of Reachy Mini, which ships in their `reachy-mini` package."""

    version: str
    """Release of `reachy-mini` on PyPI whose wheel the MJCF is unpacked from on first use, e.g.
    `1.14.0`."""
    directory: Path
    """Where descriptions are unpacked, outside the source tree; `~` is expanded."""


class SimConfig(RVABaseModel):
    """A Reachy Mini on a table, simulated in real time."""

    namespace: str
    """kickmsg namespace of the robot's topics; its clients must use the same."""
    description: ReachyMiniDescriptionConfig
    table_height: float
    """In meters. The head camera sits about 0.15 m above the table top."""
    physics_dt: float
    """Seconds of simulated time per physics step."""
    state_rate: float
    """Joint states published per second."""
    camera_fps: float
    """Frames from the head camera per second; their size and field of view are the MJCF's."""
