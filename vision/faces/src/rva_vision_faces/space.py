import math
import statistics
from collections import deque
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_schemas.vision import (
    Body,
    CameraIntrinsics,
    DepthMap,
    Face3D,
    FaceFeatures,
    Gaze,
    GazeRay,
    Hand,
    Keypoint,
    Point3D,
)
from rva_vision_faces.scene import Scene

type _Vector = npt.NDArray[np.float64]

_PATCH_RADIUS: Final = 2
"""Half the side of the square of depth pixels a point's depth is read from."""
_NEAR_PERCENTILE: Final = 25
"""Keypoints sit on the edges of limbs and fingers, where the patch also holds what is behind
them: the nearer depths are the person's."""
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""
_SHOULDERS: Final = (5, 6)
"""The shoulders among `Body.keypoints`."""
_PALM: Final = (0, 1, 5, 9, 13, 17)
"""`Hand.keypoints` on the palm: the wrist and the base of each finger."""
_HAND_REACH_M: Final = 0.1
"""How far a point of a hand may be from its palm's depth. Fingers are too thin for the depth
model, which sees the room through them: without a bound, their tips land metres behind."""
_HAND_BONES: Final = (
    (0, 1, 0.040), (1, 2, 0.045), (2, 3, 0.032), (3, 4, 0.027),
    (0, 5, 0.090), (5, 6, 0.042), (6, 7, 0.025), (7, 8, 0.022),
    (0, 9, 0.088), (9, 10, 0.046), (10, 11, 0.028), (11, 12, 0.023),
    (0, 13, 0.082), (13, 14, 0.042), (14, 15, 0.027), (15, 16, 0.022),
    (0, 17, 0.076), (17, 18, 0.033), (18, 19, 0.019), (19, 20, 0.020),
)  # fmt: skip
"""(parent, child, metres) of each bone of an average adult hand, 18.5 cm from the wrist to the
tip of the middle finger, parents first. Thumb, then index to little finger, from the wrist."""
_HAND_PARENTS: Final = {child: (parent, length) for parent, child, length in _HAND_BONES}
"""Each point of a hand but the wrist, to its parent and the bone's length."""
_STANDARD_SHOULDERS_M: Final = 0.38
"""Distance between the shoulder keypoints of the person `_HAND_BONES` are for."""
_SCALE_BOUNDS: Final = (0.8, 1.2)
"""Hands are scaled with the shoulders within these: adult hands vary by about 6%, and a far
off shoulder width is more likely a bad skeleton than a giant."""
_SHOULDER_SAMPLES: Final = 90
"""Shoulder widths kept per track, about 3 s: a person's size is their median over them."""


class Lifter:
    """Places what is seen in the camera's frame, from the depth and the camera's intrinsics.

    Keeps each tracked person's size across frames, to give their hands bones of a steady length.
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        """Set up the lifter; `intrinsics` are the camera's at the frame's resolution."""
        self._intrinsics: Final = intrinsics
        self._shoulders: dict[int, deque[float]] = {}

    def lift(self, scene: Scene, depth: DepthMap) -> Scene:
        """The scene with its keypoints, faces and gazes placed in the camera's frame."""
        intrinsics = self._intrinsics
        self._shoulders = {
            track_id: widths
            for track_id, widths in self._shoulders.items()
            if track_id in scene.bodies or track_id in scene.faces
        }
        bodies = {}
        for track_id, body in scene.bodies.items():
            widths = self._shoulders.setdefault(track_id, deque(maxlen=_SHOULDER_SAMPLES))
            bodies[track_id] = _lift_body(body, depth, intrinsics, widths)
        return scene.model_copy(
            update={
                "faces": {
                    track_id: _lift_face(face, depth, intrinsics)
                    for track_id, face in scene.faces.items()
                },
                "bodies": bodies,
                "unmatched_bodies": tuple(
                    _lift_body(body, depth, intrinsics, deque()) for body in scene.unmatched_bodies
                ),
            }
        )


def back_project(x: float, y: float, depth: DepthMap, intrinsics: CameraIntrinsics) -> Point3D:
    """The point seen at pixel (`x`, `y`) of the frame, at the depth around it."""
    ((px, py, pz),) = _unproject(
        np.array([[x, y]]), _depths(np.array([[x, y]]), depth, intrinsics), intrinsics
    )
    return Point3D(x=float(px), y=float(py), z=float(pz))


def gaze_direction(eyes: Point3D, gaze: Gaze) -> Point3D:
    """The unit vector the eyes at `eyes` look along.

    `gaze` is measured from the line from the eyes to the camera, with yaw towards the right of
    the image and pitch upwards.
    """
    distance = math.hypot(eyes.x, eyes.y, eyes.z)
    forward = np.array([-eyes.x, -eyes.y, -eyes.z]) / distance
    # The image's right and up, made square to the line to the camera.
    right = np.array([1.0, 0.0, 0.0]) - forward[0] * forward
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    yaw, pitch = math.radians(gaze.yaw), math.radians(gaze.pitch)
    direction = (
        math.cos(pitch) * math.cos(yaw) * forward
        + math.cos(pitch) * math.sin(yaw) * right
        + math.sin(pitch) * up
    )
    return Point3D(x=float(direction[0]), y=float(direction[1]), z=float(direction[2]))


def _depths(
    pixels: npt.NDArray[np.float64], depth: DepthMap, intrinsics: CameraIntrinsics
) -> npt.NDArray[np.float64]:
    """The depth around each of N pixels; patches stop at the border."""
    rows, columns = depth.metres.shape
    offsets = np.arange(-_PATCH_RADIUS, _PATCH_RADIUS + 1)
    row = np.rint(pixels[:, 1] * rows / intrinsics.height).astype(np.intp)
    column = np.rint(pixels[:, 0] * columns / intrinsics.width).astype(np.intp)
    patch_rows = np.clip(row[:, None, None] + offsets[None, :, None], 0, rows - 1)
    patch_columns = np.clip(column[:, None, None] + offsets[None, None, :], 0, columns - 1)
    patches = depth.metres[patch_rows, patch_columns].reshape(len(pixels), -1)
    return np.percentile(patches, _NEAR_PERCENTILE, axis=1).astype(np.float64)


def _unproject(
    pixels: npt.NDArray[np.float64], z: npt.NDArray[np.float64], intrinsics: CameraIntrinsics
) -> npt.NDArray[np.float64]:
    """N pixels at depths `z`, in the camera's frame."""
    return np.column_stack(
        [
            (pixels[:, 0] - intrinsics.cx) / intrinsics.fx * z,
            (pixels[:, 1] - intrinsics.cy) / intrinsics.fy * z,
            z,
        ]
    )


def _lift_face(face: FaceFeatures, depth: DepthMap, intrinsics: CameraIntrinsics) -> FaceFeatures:
    (left_x, left_y), (right_x, right_y) = (face.landmarks[iris] for iris in _IRISES)
    eyes = back_project((left_x + right_x) / 2, (left_y + right_y) / 2, depth, intrinsics)
    pixels = np.array(face.landmarks)
    landmarks = _unproject(pixels, _depths(pixels, depth, intrinsics), intrinsics)
    in_3d = Face3D(
        landmarks=tuple(Point3D(x=x, y=y, z=z) for x, y, z in landmarks.tolist()),
        gaze=GazeRay(origin=eyes, direction=gaze_direction(eyes, face.gaze)),
    )
    return face.model_copy(update={"in_3d": in_3d})


def _lift_body(
    body: Body, depth: DepthMap, intrinsics: CameraIntrinsics, shoulder_widths: deque[float]
) -> Body:
    """The body placed in the camera's frame; adds its shoulder width to `shoulder_widths`."""
    keypoints = tuple(
        None
        if k is None
        else k.model_copy(update={"position": back_project(k.x, k.y, depth, intrinsics)})
        for k in body.keypoints
    )
    left, right = (_position(keypoints, index) for index in _SHOULDERS)
    if left is not None and right is not None:
        shoulder_widths.append(math.dist((left.x, left.y, left.z), (right.x, right.y, right.z)))
    scale = (
        float(np.clip(statistics.median(shoulder_widths) / _STANDARD_SHOULDERS_M, *_SCALE_BOUNDS))
        if shoulder_widths
        else 1.0
    )

    def hand(found: Hand | None) -> Hand | None:
        return None if found is None else _lift_hand(found, depth, intrinsics, scale)

    return body.model_copy(
        update={
            "keypoints": keypoints,
            "left_hand": hand(body.left_hand),
            "right_hand": hand(body.right_hand),
        }
    )


def _position(keypoints: tuple[Keypoint | None, ...], index: int) -> Point3D | None:
    keypoint = keypoints[index] if index < len(keypoints) else None
    return None if keypoint is None else keypoint.position


def _lift_hand(hand: Hand, depth: DepthMap, intrinsics: CameraIntrinsics, scale: float) -> Hand:
    """The hand placed with bones of `_HAND_BONES`' lengths times `scale`.

    The depth places the wrist, no farther than the palm's size in the image allows; each other
    point is on its pixel's ray, a bone's length from its
    parent. Where joints between were not found, it is the bones' summed length from the nearest
    point found up the hand: the finger is taken as straight there. Of the two such points on the
    ray, one after the wrist takes the one nearer its depth, which holds on the palm; the others
    take the one that bends the finger the least, as their depth is the room's as often as the
    finger's. A point with nothing found up the hand is placed at its depth, kept near the palm's.
    """
    found = {index: k for index, k in enumerate(hand.keypoints) if k is not None}
    if not found:
        return hand
    pixels = {index: np.array([k.x, k.y]) for index, k in found.items()}
    measured = dict(
        zip(found, _depths(np.array(list(pixels.values())), depth, intrinsics), strict=True)
    )
    on_palm = [z for index, z in measured.items() if index in _PALM]
    # A hand seen without its palm, e.g. only fingertips, is anchored on all its points.
    palm = float(np.median(on_palm or list(measured.values())))
    near_palm = {
        index: min(max(z, palm - _HAND_REACH_M), palm + _HAND_REACH_M)
        for index, z in measured.items()
    }

    def at_depth(index: int) -> _Vector:
        return _unproject(pixels[index][None], np.array([near_palm[index]]), intrinsics)[0]

    positions: dict[int, _Vector] = {}

    def placed_above(index: int) -> tuple[int, float] | None:
        """The nearest point placed up the hand from `index`, and the bones' length to it."""
        reach = 0.0
        while index in _HAND_PARENTS:
            index, length = _HAND_PARENTS[index]
            reach += length
            if index in positions:
                return index, reach
        return None

    if 0 in found:
        # A bone looks shorter in the image when it is turned away, never longer: the palm's
        # bones as seen bound how far the hand can be, which the wrist's depth, read from the
        # room past it as often as not, may be beyond.
        focal = (intrinsics.fx + intrinsics.fy) / 2
        farthest = min(
            (
                focal * length * scale / pixel_length
                for parent, child, length in _HAND_BONES
                if parent == 0
                and child in found
                and (pixel_length := float(np.linalg.norm(pixels[child] - pixels[0]))) > 0
            ),
            default=math.inf,
        )
        positions[0] = _unproject(
            pixels[0][None], np.array([min(near_palm[0], farthest)]), intrinsics
        )[0]
    for _, child, _ in _HAND_BONES:
        if child not in found:
            continue
        above = placed_above(child)
        if above is None:
            positions[child] = at_depth(child)
            continue
        anchor, reach = above
        candidates = _on_ray_at(_ray(pixels[child], intrinsics), positions[anchor], reach * scale)
        before = placed_above(anchor)
        if before is None:
            positions[child] = min(candidates, key=lambda p: abs(p[2] - near_palm[child]))
            continue
        bone = positions[anchor] - positions[before[0]]
        positions[child] = max(candidates, key=lambda p: float(np.dot(p - positions[anchor], bone)))
    return Hand(
        keypoints=tuple(
            None
            if k is None
            else k.model_copy(
                update={
                    "position": Point3D(
                        x=float(positions[index][0]),
                        y=float(positions[index][1]),
                        z=float(positions[index][2]),
                    )
                }
            )
            for index, k in enumerate(hand.keypoints)
        )
    )


def _ray(pixel: _Vector, intrinsics: CameraIntrinsics) -> _Vector:
    """The unit vector from the camera through `pixel`."""
    direction = np.array(
        [
            (pixel[0] - intrinsics.cx) / intrinsics.fx,
            (pixel[1] - intrinsics.cy) / intrinsics.fy,
            1.0,
        ]
    )
    return direction / np.linalg.norm(direction)


def _on_ray_at(ray: _Vector, parent: _Vector, length: float) -> tuple[_Vector, ...]:
    """The points along `ray` at `length` from `parent`: two, or the nearest one if none is.

    None is when the bone looks longer in the image than it can be: the hand is larger than
    assumed, or a keypoint is off.
    """
    along = float(ray @ parent)
    discriminant = along**2 - float(parent @ parent) + length**2
    if discriminant < 0:
        return (along * ray,)
    root = math.sqrt(discriminant)
    return ((along - root) * ray, (along + root) * ray)
