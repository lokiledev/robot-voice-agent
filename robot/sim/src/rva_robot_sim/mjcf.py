"""What the simulation reads from the MJCF itself, rather than from Genesis."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

from rva_pydantic import RVABaseModel

HEAD_CAMERA: Final = "eye_camera"


class CameraMount(RVABaseModel):
    """A camera fixed to a body, looking along its -z axis with +y up, as MuJoCo's cameras do."""

    body: str
    """The body the camera's frame is fixed to."""
    position: tuple[float, float, float]
    """Of the camera's frame in the body's, in meters."""
    quaternion: tuple[float, float, float, float]
    """Of the camera's frame in the body's, w first."""
    width: int
    height: int
    fov_y: float
    """Vertical field of view, in degrees."""


class RobotFacts(RVABaseModel):
    actuated: tuple[str, ...]
    """The joints with an actuator, in the MJCF's order."""
    head_camera: CameraMount


def read_facts(mjcf: Path) -> RobotFacts:
    """The actuated joints and the head camera of the robot described in `mjcf`.

    Raises:
        ValueError: If the MJCF has no head camera, or places it in a way this reader does not
            follow: on a body of its own nested in the body it is fixed to, with no offset of its
            own.
    """
    root = ET.parse(mjcf).getroot()
    actuated = tuple(position.attrib["joint"] for position in root.iterfind("actuator/position"))
    return RobotFacts(actuated=actuated, head_camera=_head_camera(root))


def _head_camera(root: ET.Element) -> CameraMount:
    parents = {child: parent for parent in root.iter() for child in parent}
    camera = next(
        (element for element in root.iter("camera") if element.get("name") == HEAD_CAMERA), None
    )
    if camera is None:
        raise ValueError(f"The MJCF has no camera named {HEAD_CAMERA}")
    if "pos" in camera.attrib or "quat" in camera.attrib:
        raise ValueError(f"{HEAD_CAMERA} has an offset of its own in its body")
    frame = parents[camera]
    body = parents[frame]
    if frame.tag != "body" or body.tag != "body" or "name" not in body.attrib:
        raise ValueError(f"{HEAD_CAMERA} is not on a body nested in a named body")
    x, y, z = _floats(frame.get("pos", "0 0 0"), 3)
    w, qx, qy, qz = _floats(frame.get("quat", "1 0 0 0"), 4)
    width, height = (int(size) for size in camera.attrib["resolution"].split())
    return CameraMount(
        body=body.attrib["name"],
        position=(x, y, z),
        quaternion=(w, qx, qy, qz),
        width=width,
        height=height,
        fov_y=float(camera.attrib["fovy"]),
    )


def _floats(text: str, count: int) -> list[float]:
    values = [float(value) for value in text.split()]
    if len(values) != count:
        raise ValueError(f"Expected {count} numbers, not {text!r}")
    return values
