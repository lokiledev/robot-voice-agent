"""Rocketbox avatars posed by Rocketbox clips, and their meshes deformed, in numpy.

A skin is an avatar's rest-pose mesh with its skeleton; a clip, each bone's rotation in its parent
frame by frame, from Rocketbox's own skeleton. Both are 3ds Max Bipeds with the same bone axes, so
giving a clip's rotations to an avatar's bones, with the avatar's own bone lengths, poses it as the
clip's actor even though their rest poses differ: arms down in the clips, an A-pose in the avatars.
"""

import math
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel

type Floats = npt.NDArray[np.float64]
type Ints = npt.NDArray[np.int32]


class Skin(RVABaseModel, arbitrary_types_allowed=True):
    """An avatar's mesh in its rest pose, bound to its skeleton, in meters, in the world frame."""

    bone_names: tuple[str, ...]
    bone_parents: Ints
    """B: each bone's parent's index, before its own; -1 for the root, which is the first."""
    bone_rest: Floats
    """B×4×4: each bone's pose in the world in the rest pose, rigid."""
    positions: Floats
    """V×3: the vertices in the rest pose."""
    bone_index: Ints
    """V×4: the four bones that move each vertex most."""
    bone_weight: Floats
    """V×4: how much, summing to 1."""
    shape_names: tuple[str, ...]
    """Face shapes, e.g. `AA_VI_10_aa` (the viseme of "aa") or `AK_09_EyeBlinkLeft` (ARKit's)."""
    shape_deltas: Floats
    """K×V×3: how far each shape moves each vertex at full weight, in the rest pose."""

    @cached_property
    def local_rest(self) -> Floats:
        """B×4×4: each bone's rest pose in its parent's; the root's in the world."""
        local = self.bone_rest.copy()
        for bone in range(1, len(local)):
            parent = self.bone_rest[self.bone_parents[bone]]
            local[bone] = np.linalg.inv(parent) @ self.bone_rest[bone]
        return local

    @cached_property
    def rest_inverse(self) -> Floats:
        return np.linalg.inv(self.bone_rest).astype(np.float64)

    def shape_index(self, name: str) -> int:
        """Of a face shape in `shape_names`.

        Raises:
            ValueError: If the skin has no such shape.
        """
        return self.shape_names.index(name)


class Clip(RVABaseModel, arbitrary_types_allowed=True):
    """A Rocketbox animation as its own skeleton plays it."""

    bone_names: tuple[str, ...]
    rotations: Floats
    """T×B×4: each bone's rotation in its parent's frame, as w, x, y, z; the root's in the world."""
    root_position: Floats
    """T×3: the root's position in the world, in meters."""
    rest_root_height: float
    """Of the root above the floor in the clip skeleton's rest pose, in meters."""
    fps: float


class Motion(RVABaseModel, arbitrary_types_allowed=True):
    """A clip retargeted to one skin, starting where and as the skin stands at rest."""

    rotations: Floats
    """T×B×4: the skin's bones' rotations in their parents, w, x, y, z; the root's in the world."""
    root_position: Floats
    """T×3: the root's position in the world, from the skin's rest position at the start."""
    fps: float

    @property
    def duration(self) -> float:
        return (len(self.rotations) - 1) / self.fps


def load_skin(path: Path) -> Skin:
    with np.load(path) as data:
        return Skin(
            bone_names=tuple(str(name) for name in data["bone_names"]),
            bone_parents=data["bone_parents"].astype(np.int32),
            bone_rest=data["bone_rest"],
            positions=data["positions"],
            bone_index=data["bone_index"].astype(np.int32),
            bone_weight=data["bone_weight"],
            shape_names=tuple(str(name) for name in data["shape_names"]),
            shape_deltas=data["shape_deltas"],
        )


def load_clip(path: Path) -> Clip:
    with np.load(path) as data:
        return Clip(
            bone_names=tuple(str(name) for name in data["bone_names"]),
            rotations=data["rotations"],
            root_position=data["root_position"],
            rest_root_height=float(data["rest_root_height"]),
            fps=float(data["fps"]),
        )


def retarget(skin: Skin, clip: Clip, still: frozenset[str]) -> Motion:
    """The clip's rotations on the skin's bones, and its root's path scaled to the skin's height.

    The motion is turned and moved so that it starts facing as the skin's root does at rest, above
    the same point of the floor: a player places it from there.

    Args:
        skin: The avatar to animate.
        clip: The animation, from a skeleton with the same bone names.
        still: Bones kept in their rest pose with all their descendants, e.g. the head's, whose
            face bones the clips do not animate but whose rest pose is each avatar's own.

    Raises:
        ValueError: If the skin has bones the clip does not, other than still ones.
    """
    kept = _descendants(skin, still)
    column = {name: index for index, name in enumerate(clip.bone_names)}
    missing = [name for name in skin.bone_names if name not in column and name not in kept]
    if missing:
        raise ValueError(f"The clip has no bones {missing}")
    rest_local = rest(skin).rotations[0]
    frames = len(clip.rotations)
    rotations = np.empty((frames, len(skin.bone_names), 4))
    for bone, name in enumerate(skin.bone_names):
        if name in kept:
            rotations[:, bone] = rest_local[bone]
        else:
            rotations[:, bone] = clip.rotations[:, column[name]]
    rest_root = skin.bone_rest[0]
    scale = rest_root[2, 3] / clip.rest_root_height
    start_heading = heading(_matrices(rotations[0, 0]) @ rest_root[:3, :3].T)
    turn = _yaw_quaternion(-start_heading)
    rotations[:, 0] = _multiply(turn, rotations[:, 0])
    offset = (clip.root_position - clip.root_position[0] * (1.0, 1.0, 0.0)) * scale
    root_position = offset @ _matrices(turn).T + rest_root[:3, 3] * (1.0, 1.0, 0.0)
    return Motion(rotations=rotations, root_position=root_position, fps=clip.fps)


def rest(skin: Skin) -> Motion:
    """A motion of one frame that holds the skin in its rest pose: still, in an A-pose."""
    rotations = np.stack([_quaternion(local[:3, :3]) for local in skin.local_rest])
    return Motion(rotations=rotations[None], root_position=skin.bone_rest[None, 0, :3, 3], fps=1.0)


def aimed(
    skin: Skin, motion: Motion, aims: Sequence[tuple[str, str, tuple[float, float, float]]]
) -> Motion:
    """The motion with some bones held pointing one way all along, e.g. an arm that carries.

    Args:
        skin: The avatar the motion is retargeted to.
        motion: What the other bones keep playing.
        aims: Each a bone, the bone at its end, and the way from the one to the other, in the
            skin's frame at rest: it faces -y, +x is its left. Never the root; a bone comes
            after its parent when both are aimed. The way holds while the bone's parent stands as at rest, and
            turns with it.
    """
    rest_world = skin.bone_rest[:, :3, :3].copy()
    rotations = motion.rotations.copy()
    for name, end, way in aims:
        bone, tip = skin.bone_names.index(name), skin.bone_names.index(end)
        along = skin.bone_rest[tip, :3, 3] - skin.bone_rest[bone, :3, 3]
        rest_world[bone] = _turn(along, np.array(way)) @ rest_world[bone]
        rotations[:, bone] = _quaternion(rest_world[skin.bone_parents[bone]].T @ rest_world[bone])
    return Motion(rotations=rotations, root_position=motion.root_position, fps=motion.fps)


def truncate(motion: Motion, seconds: float) -> Motion:
    """Its first `seconds`, or all of it if shorter."""
    frames = min(len(motion.rotations), math.floor(seconds * motion.fps) + 1)
    return Motion(
        rotations=motion.rotations[:frames],
        root_position=motion.root_position[:frames],
        fps=motion.fps,
    )


def sample(motion: Motion, time: float) -> tuple[Floats, Floats]:
    """The bones' rotations and the root's position `time` seconds in, clamped to the motion."""
    position = min(max(time, 0.0), motion.duration) * motion.fps
    before = min(int(position), len(motion.rotations) - 1)
    after = min(before + 1, len(motion.rotations) - 1)
    weight = position - before
    rotations = nlerp(motion.rotations[before], motion.rotations[after], weight)
    root = (1 - weight) * motion.root_position[before] + weight * motion.root_position[after]
    return rotations, root


def pose(skin: Skin, rotations: Floats, root_position: Floats, placement: Floats) -> Floats:
    """B×4×4: each bone's pose in the world, the skin's bone lengths turned by `rotations`.

    Args:
        skin: The avatar.
        rotations: B×4, each bone's rotation in its parent; the root's in the placement's frame.
        root_position: Of the root, in the placement's frame.
        placement: 4×4, where the avatar's frame is in the world.
    """
    offsets = skin.local_rest[:, :3, 3]
    turns = _matrices(rotations)
    bones = np.empty((len(skin.bone_names), 4, 4))
    bones[0] = placement @ _rigid(turns[0], root_position)
    for bone in range(1, len(bones)):
        parent = bones[skin.bone_parents[bone]]
        bones[bone] = parent @ _rigid(turns[bone], offsets[bone])
    return bones


def deform(skin: Skin, bones: Floats, shape_weights: Floats) -> Floats:
    """V×3: the vertices in the world, the face shapes added in the rest pose, then skinned.

    Args:
        skin: The avatar.
        bones: B×4×4, from `pose`.
        shape_weights: K, one per face shape, from 0 to 1.
    """
    rest = skin.positions + np.tensordot(shape_weights, skin.shape_deltas, axes=1)
    transforms = (bones @ skin.rest_inverse)[:, :3, :]
    # Blending the four bones' transforms, then moving each vertex once, is six times faster than
    # moving it by each bone and blending the results.
    blended = np.einsum("vk,vkij->vij", skin.bone_weight, transforms[skin.bone_index])
    return np.einsum("vij,vj->vi", blended[:, :, :3], rest) + blended[:, :, 3]


def displacement(skin: Skin, motion: Motion, time: float) -> Floats:
    """4×4: how the motion moves and turns the avatar on the floor in its first `time` seconds.

    The next motion is placed from there by composing its placement with this.
    """
    rotations, root = sample(motion, time)
    turn = heading(_matrices(rotations[0]) @ skin.bone_rest[0, :3, :3].T)
    x, y, _ = root - motion.root_position[0]
    return placement(x, y, turn)


def heading(rotation: Floats) -> float:
    """The yaw of a 3×3 rotation, about +z, in radians: where it turns +x on the floor."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


def nlerp(start: Floats, end: Floats, weight: float) -> Floats:
    """Quaternions between `start` and `end` along the shorter way, normalized."""
    end = np.where(np.sum(start * end, axis=-1, keepdims=True) < 0, -end, end)
    mixed = (1 - weight) * start + weight * end
    return mixed / np.linalg.norm(mixed, axis=-1, keepdims=True)


def placement(x: float, y: float, yaw: float) -> Floats:
    """4×4: on the floor at x, y, turned by `yaw` radians about +z."""
    transform = np.eye(4)
    transform[:3, :3] = _matrices(_yaw_quaternion(yaw))
    transform[:2, 3] = (x, y)
    return transform


def _descendants(skin: Skin, roots: frozenset[str]) -> frozenset[str]:
    kept: set[str] = set()
    for bone, name in enumerate(skin.bone_names):
        parent = skin.bone_parents[bone]
        if name in roots or (parent >= 0 and skin.bone_names[parent] in kept):
            kept.add(name)
    return frozenset(kept)


def _turn(start: Floats, end: Floats) -> Floats:
    """The smallest 3×3 rotation that points `start` the way of `end`; they are not opposite."""
    start, end = start / np.linalg.norm(start), end / np.linalg.norm(end)
    x, y, z = np.cross(start, end)
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + skew + skew @ skew / (1 + start @ end)


def _rigid(rotation: Floats, translation: Floats) -> Floats:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _matrices(quaternions: Floats) -> Floats:
    """…×3×3 rotations of …×4 unit quaternions, w, x, y, z."""
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def _quaternion(rotation: Floats) -> Floats:
    """The unit quaternion, w, x, y, z, of a 3×3 rotation."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = (s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4)
    return np.array(q)


def _yaw_quaternion(yaw: float) -> Floats:
    return np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def _multiply(a: Floats, b: Floats) -> Floats:
    """Hamilton products of quaternions, w, x, y, z, broadcast."""
    aw, ax, ay, az = np.moveaxis(np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.broadcast_to(b, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )
