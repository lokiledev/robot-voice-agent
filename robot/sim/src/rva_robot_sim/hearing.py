"""What the simulated robot's microphones hear: where the person speaking is.

The array is in the head, at the camera, as Reachy Mini's microphones are: it turns with the
head. Its frame has x along the camera's view, y to the image's left and z up the image, so the
azimuths it reports grow to the left, as the reSpeaker's do lying flat facing ahead.
"""

import math

import numpy as np

from rva_robot_sim.skinning import Floats


def heard_azimuth(camera: Floats, point: Floats) -> float:
    """The azimuth, in radians, at which the array hears a sound made at `point`.

    Args:
        camera: 4×4, the camera's pose in the world, looking along its -z axis with +y up, as
            Genesis' and MuJoCo's cameras do.
        point: 3, where the sound is made, in the world.
    """
    x, _, z = camera[:3, :3].T @ (np.asarray(point) - camera[:3, 3])
    # Ahead is -z, the left -x.
    return math.atan2(-x, -z)
