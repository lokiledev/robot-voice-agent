"""The head controller in rerun: the robot's meshes where its joints put them, and the target."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import rerun as rr

from rva_pydantic import RVABaseModel
from rva_robot_kinematics.head import pose_matrix
from rva_robot_schemas.head import HeadPose

_BASE: Final = "body_foot_3dprint"
_BODY: Final = "body_down_3dprint"
"""Turned by `yaw_body`, about its z axis."""
_HEAD: Final = "xl_330"
"""Carries the head site, the camera and the antennas."""
_HEAD_SITE: Final = "head"
_ROOT: Final = "robot"
type Part = Literal["base", "body", "head"]


class Mesh(RVABaseModel, arbitrary_types_allowed=True):
    part: Part
    """The rigid part it moves with."""
    file: Path
    pose: npt.NDArray[np.float64]
    """4×4 pose in its part's frame."""
    rgba: tuple[float, float, float, float]


class RobotMeshes(RVABaseModel, arbitrary_types_allowed=True):
    """The robot's visual meshes as three rigid parts.

    The Stewart platform's arms and rods, and the antennas' turning, are left out.
    """

    meshes: tuple[Mesh, ...]
    body: npt.NDArray[np.float64]
    """4×4 pose of the body's frame in the base's, with the body's yaw at zero."""
    head_site: npt.NDArray[np.float64]
    """4×4 pose of the head site in the head's frame."""


def read_meshes(mjcf: Path) -> RobotMeshes:
    """The meshes of the base, the body and the head of the robot `mjcf` describes."""
    root = ET.parse(mjcf).getroot()
    compiler = root.find("compiler")
    assert compiler is not None, "every MJCF of Reachy Mini has a compiler, naming its meshdir"
    directory = mjcf.parent / compiler.attrib["meshdir"]
    files = {
        element.get("name", Path(element.attrib["file"]).stem): directory / element.attrib["file"]
        for element in root.iterfind("asset/mesh")
    }
    colours = {
        element.attrib["name"]: _floats(element.get("rgba", "0.7 0.7 0.7 1"))
        for element in root.iterfind("asset/material")
    }
    bodies = {
        element.attrib["name"]: element for element in root.iter("body") if "name" in element.attrib
    }
    meshes = [
        *_geoms(bodies[_BASE], "base", np.eye(4), files, colours, descend=False),
        *_geoms(bodies[_BODY], "body", np.eye(4), files, colours, descend=False),
        *_geoms(bodies[_HEAD], "head", np.eye(4), files, colours, descend=True),
    ]
    site = next(site for site in bodies[_HEAD].iter("site") if site.get("name") == _HEAD_SITE)
    return RobotMeshes(meshes=tuple(meshes), body=_local(bodies[_BODY]), head_site=_local(site))


class HeadView:
    """Shows, in rerun, the robot as the controller believes it is, and where the head should be.

    The robot's meshes are logged once; then each estimate moves the body and the head.
    """

    def __init__(self, meshes: RobotMeshes) -> None:
        self._meshes: Final = meshes
        self._site_inverse: Final = np.linalg.inv(meshes.head_site)
        rr.log(_ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for index, mesh in enumerate(meshes.meshes):
            path = f"{_ROOT}/{mesh.part}/mesh_{index}"
            rr.log(path, _transform(mesh.pose), static=True)
            rr.log(path, rr.Asset3D(path=mesh.file, albedo_factor=mesh.rgba), static=True)

    def show_estimate(self, timestamp: float, body_yaw: float, head: HeadPose) -> None:
        """Move the body and the head where the joints put them."""
        rr.set_time("time", duration=timestamp)
        rr.log(f"{_ROOT}/body", _transform(self._meshes.body @ _rotation_z(body_yaw)))
        rr.log(f"{_ROOT}/head", _transform(pose_matrix(head) @ self._site_inverse))
        rr.log(f"{_ROOT}/head_estimate", _transform(pose_matrix(head)), rr.TransformAxes3D(0.05))

    def show_target(self, timestamp: float, target: HeadPose) -> None:
        rr.set_time("time", duration=timestamp)
        rr.log(f"{_ROOT}/head_target", _transform(pose_matrix(target)), rr.TransformAxes3D(0.08))

    def show_gaze(self, timestamp: float, point: npt.NDArray[np.float64], target: HeadPose) -> None:
        """Show a gaze target's point, in the base frame, and the head pose that looks at it."""
        self.show_target(timestamp, target)
        rr.log(f"{_ROOT}/gaze_target", rr.Points3D([point], radii=0.02))


def _geoms(
    body: ET.Element,
    part: Part,
    pose: npt.NDArray[np.float64],
    files: dict[str, Path],
    colours: dict[str, tuple[float, ...]],
    *,
    descend: bool,
) -> list[Mesh]:
    """The meshes of `body`, posed in its part's frame.

    With `descend`, those of its children too, at their joints' zero.
    """
    meshes = [
        Mesh(
            part=part,
            file=files[geom.attrib["mesh"]],
            pose=pose @ _local(geom),
            rgba=_rgba(colours.get(geom.get("material", ""), (0.7, 0.7, 0.7, 1.0))),
        )
        for geom in body.findall("geom")
        if geom.get("type") == "mesh" and geom.get("class") == "visual"
    ]
    if descend:
        for child in body.findall("body"):
            meshes += _geoms(child, part, pose @ _local(child), files, colours, descend=True)
    return meshes


def _local(element: ET.Element) -> npt.NDArray[np.float64]:
    """An MJCF element's pose in its parent's frame, from its `pos` and `quat`."""
    w, x, y, z = _floats(element.get("quat", "1 0 0 0"))
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = _floats(element.get("pos", "0 0 0"))
    return matrix


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    matrix = np.eye(4)
    cos, sin = math.cos(angle), math.sin(angle)
    matrix[:2, :2] = ((cos, -sin), (sin, cos))
    return matrix


def _transform(matrix: npt.NDArray[np.float64]) -> rr.Transform3D:
    return rr.Transform3D(translation=matrix[:3, 3], mat3x3=matrix[:3, :3])


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


def _rgba(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    red, green, blue, alpha = values
    return (red, green, blue, alpha)
