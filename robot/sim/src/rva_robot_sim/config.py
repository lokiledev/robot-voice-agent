from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_sim.hdri import HdriConfig
from rva_robot_sim.people import AvatarsConfig, PlacedPerson


class ReachyMiniDescriptionConfig(RVABaseModel):
    """Pollen Robotics' MJCF of Reachy Mini, which ships in their `reachy-mini` package."""

    version: str
    """Release of `reachy-mini` on PyPI whose wheel the MJCF is unpacked from on first use, e.g.
    `1.14.0`."""
    directory: Path
    """Where descriptions are unpacked, outside the source tree; `~` is expanded."""


class NyxRenderer(RVABaseModel):
    """Nyx, Genesis' GPU path tracer: lit and shaded like a photograph.

    About 4 ms a frame on an RTX 4090. Needs an NVIDIA GPU with ray tracing, and an x86-64 machine.
    """

    kind: Literal["nyx"] = "nyx"
    samples_per_pixel: int = Field(gt=0)
    """Light paths traced per pixel each frame, before denoising."""
    environment: HdriConfig
    """Seen around the world, and lighting it with the sun."""


class Rasterizer(RVABaseModel):
    """Genesis' OpenGL rasterizer: flat shading, but no ray-tracing GPU needed."""

    kind: Literal["rasterizer"] = "rasterizer"


type CameraRenderer = Annotated[NyxRenderer | Rasterizer, Field(discriminator="kind")]


class SimConfig(RVABaseModel):
    """A Reachy Mini on a table, simulated in real time."""

    namespace: str
    """kickmsg namespace of the robot's topics; its clients must use the same."""
    description: ReachyMiniDescriptionConfig
    table_height: float
    """In meters. The head camera sits about 0.15 m above the table top."""
    avatars: AvatarsConfig
    people: tuple[PlacedPerson, ...]
    """Standing still around the robot, for the camera to see."""
    physics_dt: float
    """Seconds of simulated time per physics step."""
    state_rate: float
    """Joint states published per second."""
    camera_fps: float
    """Frames from the head camera per second; their size and field of view are the MJCF's."""
    camera_renderer: CameraRenderer
