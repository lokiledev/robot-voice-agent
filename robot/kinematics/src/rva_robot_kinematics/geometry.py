"""The geometry of Reachy Mini's head, read from its MJCF: the description the simulation runs."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel

HEAD_SITE: Final = "head"
"""The site whose pose a head pose gives."""
_CAMERA_SITE: Final = "camera_optical"
"""The head camera's optical frame, as OpenCV has it, rather than MuJoCo's camera frame."""
_BODY_JOINT: Final = "yaw_body"
_MOTORS: Final = tuple(f"stewart_{motor}" for motor in range(1, 7))
_ROTATIONS: Final = ("axisangle", "euler", "xyaxes", "zaxis")
"""Ways of orienting a frame in MJCF other than `quat`, which this reader does not follow."""


class StewartLeg(RVABaseModel, arbitrary_types_allowed=True):
    """A motor of the Stewart platform, its arm, and the rod from the arm's tip to the head."""

    joint: str
    """The motor's joint, e.g. `stewart_1`."""
    motor: npt.NDArray[np.float64]
    """4×4 pose of the motor in the body's frame. The arm turns about its z axis, and its tip is
    at `HeadGeometry.arm_length` along its x axis when the joint is at zero."""
    anchor: npt.NDArray[np.float64]
    """Where the rod meets the head, in the head's frame, in meters."""
    branch: Literal[-1, 1]
    """Which of the two arm angles that put the tip at rod length from the anchor the robot is
    assembled in: the anchor's direction around the motor's axis, plus (1) or minus (-1) the angle
    between it and the arm."""
    range: tuple[float, float]
    """Lowest and highest angle of the joint, in radians."""


class HeadGeometry(RVABaseModel, arbitrary_types_allowed=True):
    """The body's yaw joint and the Stewart platform the head sits on."""

    body_joint: str
    """The joint that turns the body, and the head with it, about the vertical."""
    body: npt.NDArray[np.float64]
    """4×4 pose of the body's frame in the base's when the body joint is at zero. The body turns
    about the frame's z axis."""
    body_range: tuple[float, float]
    """Lowest and highest angle of the body joint, in radians."""
    head: npt.NDArray[np.float64]
    """4×4 pose of the head frame, the `head` site, in the body's frame with every joint at zero."""
    camera: npt.NDArray[np.float64]
    """4×4 pose of the head camera's optical frame in the head frame: x towards the right of the
    image, y down and z forward along its axis, as OpenCV has it."""
    arm_length: float
    """From a motor's axis to the ball joint at its arm's tip, in meters."""
    rod_length: float
    """Between the ball joints at the two ends of a rod, in meters."""
    legs: tuple[StewartLeg, ...]


def read_head_geometry(mjcf: Path) -> HeadGeometry:
    """The geometry of the head of the Reachy Mini that `mjcf` describes.

    The MJCF builds the Stewart platform as a tree: the head hangs from the rod of one motor, and
    the rods of the others are connected to it by equality constraints. The pose with every joint
    at zero closes these loops; it tells which way each arm is bent.

    Raises:
        ValueError: If the MJCF lacks the body joint, a motor, the head site or the camera's
            optical site, or builds them in a way this reader does not follow: a camera off the
            head's body, frames oriented otherwise than by quaternions, motors not mounted on the
            turning body, joints offset from their body's origin or turning about another axis
            than z, arms or rods of different lengths, or loops the pose with every joint at zero
            does not close.
    """
    root = ET.parse(mjcf).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    rest = _rest_poses(root)
    body_joint = _hinge(root, _BODY_JOINT)
    body = parents[body_joint]
    head_site = _named(root, "site", HEAD_SITE)
    head_link = parents[head_site]
    head_in_link = _local(head_site)
    camera_site = _named(root, "site", _CAMERA_SITE)
    if parents[camera_site] is not head_link:
        raise ValueError(f"{_CAMERA_SITE} is not on the body the {HEAD_SITE} site is on")
    head_at_rest = np.linalg.inv(rest[body]) @ rest[head_site]
    connections = _connections(root)
    legs = []
    arm_lengths = []
    rod_lengths = []
    for name in _MOTORS:
        joint = _hinge(root, name)
        horn = parents[joint]
        if parents[horn] is not body:
            raise ValueError(f"{name} is not mounted on the body {_BODY_JOINT} turns")
        rod = _rod(horn, name)
        arm_x, arm_y, arm_z = _local(rod)[:3, 3]
        if arm_x <= 0 or abs(arm_y) > 1e-6:
            raise ValueError(f"The arm of {name} does not point along x at zero")
        rod_end, anchor_in_link = _rod_end(rod, head_link, connections, name)
        arm_lengths.append(float(arm_x))
        rod_lengths.append(float(np.linalg.norm(rod_end)))
        motor = _local(horn) @ _translation(0.0, 0.0, arm_z)
        anchor = (np.linalg.inv(head_in_link) @ np.append(anchor_in_link, 1.0))[:3]
        anchor_at_rest = np.linalg.inv(motor) @ head_at_rest @ np.append(anchor, 1.0)
        if abs(math.dist(anchor_at_rest[:3], (arm_x, 0.0, 0.0)) - rod_lengths[-1]) > 1e-5:
            raise ValueError(f"With every joint at zero, the rod of {name} misses the head")
        legs.append(
            StewartLeg(
                joint=name,
                motor=motor,
                anchor=anchor,
                # At rest, the arm is at zero: the anchor is on its side opposite the branch.
                branch=-1 if math.atan2(anchor_at_rest[1], anchor_at_rest[0]) > 0 else 1,
                range=_range(joint),
            )
        )
    for part, lengths in (("arms", arm_lengths), ("rods", rod_lengths)):
        if max(lengths) - min(lengths) > 1e-6:
            raise ValueError(f"The {part} have different lengths: {lengths}")
    return HeadGeometry(
        body_joint=_BODY_JOINT,
        body=rest[body],
        body_range=_range(body_joint),
        head=head_at_rest,
        camera=np.linalg.inv(head_in_link) @ _local(camera_site),
        arm_length=arm_lengths[0],
        rod_length=rod_lengths[0],
        legs=tuple(legs),
    )


def _rest_poses(root: ET.Element) -> dict[ET.Element, npt.NDArray[np.float64]]:
    """The pose of every body and site in the base's frame, with every joint at zero."""
    poses: dict[ET.Element, npt.NDArray[np.float64]] = {}
    pending = [(child, np.eye(4)) for child in root.iterfind("worldbody/*")]
    while pending:
        element, parent = pending.pop()
        if element.tag not in ("body", "site"):
            continue
        poses[element] = parent @ _local(element)
        pending.extend((child, poses[element]) for child in element)
    return poses


def _local(element: ET.Element) -> npt.NDArray[np.float64]:
    """The pose of a body or site in its parent body's frame."""
    oriented = [attribute for attribute in _ROTATIONS if attribute in element.attrib]
    if oriented:
        raise ValueError(f"{element.get('name')} is oriented by {oriented[0]}, not by quat")
    quaternion = _floats(element.get("quat", "1 0 0 0"), 4)
    # MuJoCo normalizes quaternions, which the MJCF gives to six digits.
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    pose = np.eye(4)
    pose[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    pose[:3, 3] = _floats(element.get("pos", "0 0 0"), 3)
    return pose


def _translation(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    pose = np.eye(4)
    pose[:3, 3] = (x, y, z)
    return pose


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    element = next((e for e in root.iter(tag) if e.get("name") == name), None)
    if element is None:
        raise ValueError(f"The MJCF has no {tag} named {name}")
    return element


def _hinge(root: ET.Element, name: str) -> ET.Element:
    """A hinge joint at its body's origin, turning about the body's z axis."""
    joint = _named(root, "joint", name)
    if joint.get("type", "hinge") != "hinge" or "pos" in joint.attrib:
        raise ValueError(f"{name} is not a hinge at its body's origin")
    if tuple(_floats(joint.get("axis", "0 0 1"), 3)) != (0.0, 0.0, 1.0):
        raise ValueError(f"{name} does not turn about its body's z axis")
    return joint


def _range(joint: ET.Element) -> tuple[float, float]:
    lower, upper = _floats(joint.attrib["range"], 2)
    return float(lower), float(upper)


def _rod(horn: ET.Element, motor: str) -> ET.Element:
    """The body of a motor that turns on a ball joint at its origin: the rod."""
    for child in horn.iterfind("body"):
        balls = [joint for joint in child.iterfind("joint") if joint.get("type") == "ball"]
        if balls:
            if any("pos" in ball.attrib for ball in balls):
                raise ValueError(f"The rod of {motor} turns about another point than its origin")
            return child
    raise ValueError(f"{motor} carries no rod on a ball joint")


def _connections(root: ET.Element) -> dict[str, ET.Element]:
    """For each site an equality constraint connects, the site it is connected to."""
    sites = {site.get("name", ""): site for site in root.iter("site")}
    connections = {}
    for connect in root.iterfind("equality/connect"):
        first, second = connect.get("site1", ""), connect.get("site2", "")
        if first in sites and second in sites:
            connections[first] = sites[second]
            connections[second] = sites[first]
    return connections


def _rod_end(
    rod: ET.Element, head_link: ET.Element, connections: dict[str, ET.Element], motor: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Where the rod meets the head: in the rod's frame, and in the head link's.

    Either the head link hangs from the rod, or a site at the rod's end is connected to a site on
    the head link.
    """
    if any(child is head_link for child in rod):
        return _local(head_link)[:3, 3], np.zeros(3)
    for site in rod.iterfind("site"):
        other = connections.get(site.get("name", ""))
        if other is not None and any(child is other for child in head_link):
            return _local(site)[:3, 3], _local(other)[:3, 3]
    raise ValueError(f"The rod of {motor} is neither connected to the head nor carries it")


def _floats(text: str, count: int) -> npt.NDArray[np.float64]:
    values = np.array([float(value) for value in text.split()])
    if len(values) != count:
        raise ValueError(f"Expected {count} numbers, not {text!r}")
    return values
