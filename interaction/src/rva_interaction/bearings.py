"""Directions around the camera's vertical axis, in degrees right of its optical axis.

A face's comes from where its box is in the image, so it needs no depth; a sound's from the
microphone array's frame turned into the camera's. Both leave out the height, which a flat array
cannot tell.
"""

import math

from rva_schemas.frames import StaticTransforms
from rva_schemas.sound import HeardDirection
from rva_schemas.vision import BoundingBox, CameraIntrinsics


def face_bearing(box: BoundingBox, intrinsics: CameraIntrinsics) -> float:
    """Of the centre of a face's box, in pixels of the image `intrinsics` describe."""
    return math.degrees(math.atan2(box.x + box.width / 2 - intrinsics.cx, intrinsics.fx))


def speech_bearing(
    direction: HeardDirection, camera_frame: str, transforms: StaticTransforms
) -> float | None:
    """Of the speech heard, `None` when none is.

    Raises:
        MissingTransformError: If no transform links the array's frame to the camera's.
    """
    if direction.speech_azimuth is None:
        return None
    x, _, z = transforms.rotate(
        (math.cos(direction.speech_azimuth), math.sin(direction.speech_azimuth), 0.0),
        source=direction.frame,
        target=camera_frame,
    )
    return math.degrees(math.atan2(x, z))


def view_bearings(intrinsics: CameraIntrinsics) -> tuple[float, float]:
    """Of the image's left and right edges."""
    return (
        math.degrees(math.atan2(-intrinsics.cx, intrinsics.fx)),
        math.degrees(math.atan2(intrinsics.width - intrinsics.cx, intrinsics.fx)),
    )


def apart(first: float, second: float) -> float:
    """How far apart two bearings are, in degrees from 0 to 180, the short way round."""
    return abs((first - second + 180) % 360 - 180)
