"""People in the simulated world: Microsoft's Rocketbox avatars, and where they stand."""

import math
from pathlib import Path

from rva_pydantic import RVABaseModel
from rva_robot_sim.skinning import Floats, placement


class AvatarsConfig(RVABaseModel):
    """Microsoft's Rocketbox avatars and animations, MIT-licensed, downloaded on first use."""

    commit: str
    """Commit of github.com/microsoft/Microsoft-Rocketbox the avatars are taken from."""
    directory: Path
    """Where the baked avatars and clips are kept, outside the source tree; `~` is expanded."""
    blender: Path
    """Blender's executable, which bakes each avatar and clip once."""


def yaw_deg(position: tuple[float, float], turned_deg: float) -> float:
    """The rotation about the vertical of an avatar, which faces -y at rest, placed as given.

    Args:
        position: Where they stand, in meters, in the robot's frame: it faces +x, +y is its left.
        turned_deg: How far they are turned from facing the robot, in degrees, counterclockwise
            seen from above; 0 faces it.
    """
    x, y = position
    towards_robot = math.degrees(math.atan2(-y, -x))
    return towards_robot + 90.0 + turned_deg


def standing(position: tuple[float, float], turned_deg: float) -> Floats:
    """4×4: where an avatar stands on the floor, placed as `yaw_deg` takes it."""
    return placement(*position, math.radians(yaw_deg(position, turned_deg)))
