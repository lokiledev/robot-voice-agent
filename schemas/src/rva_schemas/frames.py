import math
from collections.abc import Sequence
from typing import Final

from rva_pydantic import RVABaseModel

type Vector = tuple[float, float, float]
type _Matrix = tuple[Vector, Vector, Vector]


class FrameTransform(RVABaseModel):
    """Where a child link is in its parent's frame, as a URDF joint's origin says it."""

    parent: str
    child: str
    xyz: Vector
    """The child's origin in the parent's frame, in metres."""
    rpy: Vector
    """The child's orientation in the parent's frame: roll, pitch and yaw in radians, about the
    parent's fixed x, y and z axes in that order, as in URDF."""

    def rotate(self, vector: Vector) -> Vector:
        """`vector`, a direction in the child's frame, in the parent's."""
        return _times(self._rotation(), vector)

    def unrotate(self, vector: Vector) -> Vector:
        """`vector`, a direction in the parent's frame, in the child's."""
        rows = self._rotation()
        # A rotation's inverse is its transpose.
        return _times(
            (
                (rows[0][0], rows[1][0], rows[2][0]),
                (rows[0][1], rows[1][1], rows[2][1]),
                (rows[0][2], rows[1][2], rows[2][2]),
            ),
            vector,
        )

    def _rotation(self) -> _Matrix:
        """Rz(yaw) @ Ry(pitch) @ Rx(roll), row by row."""
        roll, pitch, yaw = self.rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )


class CameraPose(RVABaseModel):
    """Where a moving camera's optical frame was at one moment, in a frame fixed to the world.

    On a robot, its base frame; the optical frame has x right, y down and z ahead, as OpenCV has it.
    """

    position: Vector
    """The camera's origin, in metres."""
    rotation: tuple[Vector, Vector, Vector]
    """The rotation from the optical frame to the fixed one, row by row: its columns are the
    optical frame's axes in the fixed one."""

    def to_fixed(self, point: Vector) -> Vector:
        """`point`, in the optical frame, in the fixed one."""
        x, y, z = _times(self.rotation, point)
        px, py, pz = self.position
        return (x + px, y + py, z + pz)

    def turn_since(self, earlier: "CameraPose") -> tuple[Vector, Vector, Vector]:
        """The camera's own turn since `earlier`, row by row.

        It takes a direction in the optical frame `earlier` to the same direction in this one.
        """
        # Into the fixed frame by `earlier`'s rotation, out of it by this one's transpose.
        mine, theirs = self.rotation, earlier.rotation

        def entry(i: int, j: int) -> float:
            return sum(mine[k][i] * theirs[k][j] for k in range(3))

        return (
            (entry(0, 0), entry(0, 1), entry(0, 2)),
            (entry(1, 0), entry(1, 1), entry(1, 2)),
            (entry(2, 0), entry(2, 1), entry(2, 2)),
        )


class MissingTransformError(Exception):
    """No transform is known between two links."""

    def __init__(self, source: str, target: str) -> None:
        super().__init__(f"No transform between {source} and {target}")


class StaticTransforms:
    """Links fixed to each other, until the robot's description and joints tell where they are.

    Only a transform given between the two links themselves is found, either way round: none is
    chained through a third link.
    """

    def __init__(self, transforms: Sequence[FrameTransform]) -> None:
        self._transforms: Final = {(t.parent, t.child): t for t in transforms}

    def rotate(self, vector: Vector, *, source: str, target: str) -> Vector:
        """`vector`, a direction in the `source` link's frame, in the `target` link's.

        Raises:
            MissingTransformError: If no transform between the two links is known.
        """
        if source == target:
            return vector
        if (transform := self._transforms.get((target, source))) is not None:
            return transform.rotate(vector)
        if (transform := self._transforms.get((source, target))) is not None:
            return transform.unrotate(vector)
        raise MissingTransformError(source, target)

    def connects(self, first: str, second: str) -> bool:
        """Whether a direction can be taken from either link's frame to the other's."""
        return (
            first == second
            or (first, second) in self._transforms
            or (second, first) in self._transforms
        )


def _times(rows: _Matrix, vector: Vector) -> Vector:
    x, y, z = vector
    return (
        rows[0][0] * x + rows[0][1] * y + rows[0][2] * z,
        rows[1][0] * x + rows[1][1] * y + rows[1][2] * z,
        rows[2][0] * x + rows[2][1] * y + rows[2][2] * z,
    )
