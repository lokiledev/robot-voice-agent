import math
import statistics
from collections import deque
from collections.abc import Mapping
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
from rva_vision_faces.hand_model import Params, Side, fit
from rva_vision_faces.scene import Scene

_PATCH_RADIUS: Final = 2
"""Half the side of the square of depth pixels a point's depth is read from."""
_NEAR_PERCENTILE: Final = 25
"""Keypoints sit on the edges of limbs and fingers, where the patch also holds what is behind
them: the nearer depths are the person's."""
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""
_ARM_WRISTS: Final = (9, 10)
"""The left and right wrists among `Body.keypoints`."""
_PALM: Final = (0, 1, 5, 9, 13, 17)
"""`Hand.keypoints` on the palm: the wrist and the base of each finger."""
_SIDES: Final[tuple[Side, Side]] = ("left", "right")
_SIZES_KEPT_S: Final = 3.0
"""How long the fitted sizes of a track's hands are kept: their median is the size its hands are
held near."""
_ARM_M: Final = 0.6
"""How far in front of its owner's face a hand may be: an arm's length. Read from the depth, a
raised hand's wrist came out 0.9 m in front of the face, and the head swung to look at it."""
_FIT_KEPT_S: Final = 0.5
"""How long a hand's last fit is kept while it is not seen: a hand missed for a frame starts again
from where it was, not from scratch."""


class Lifter:
    """Places what is seen in the camera's frame, from the depth and the camera's intrinsics.

    Every face is placed, with its gaze, and every skeleton, with its hands. The main user's hands
    are a kinematic hand fitted to their keypoints, each from its fit in the last frames: gestures
    are read from the main user alone, and a hand's fit is paid per hand. Everyone else's are flat
    at their wrist's depth, which is enough to tell where a hand is.
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        """Set up the lifter; `intrinsics` are the camera's at the frame's resolution."""
        self._intrinsics: Final = intrinsics
        self._hands: dict[tuple[int, Side], tuple[Params, float]] = {}
        """The last fit of each person's hands, and the time of its frame."""
        self._sizes: dict[int, deque[tuple[float, float]]] = {}
        """The latest fitted sizes of each person's hands, each with the time of its frame."""

    def lift(self, scene: Scene, depth: DepthMap, main_user: int | None, time: float) -> Scene:
        """The scene placed in the camera's frame: every face, gaze, skeleton and hand.

        Args:
            scene: What was seen in a frame, in its pixels.
            depth: The frame's depth.
            main_user: The track whose hands are fitted, if it has a skeleton.
            time: When the frame was taken, in seconds on a monotonic clock.
        """
        intrinsics = self._intrinsics
        self._sizes = {id_: sizes for id_, sizes in self._sizes.items() if id_ in scene.bodies}
        self._hands = {
            key: (params, fitted_at)
            for key, (params, fitted_at) in self._hands.items()
            if time - fitted_at < _FIT_KEPT_S
        }
        faces = {
            track_id: _lift_face(face, depth, intrinsics) for track_id, face in scene.faces.items()
        }
        # How near each person's hands may be: an arm's length in front of their eyes.
        nearest = {
            track_id: face.in_3d.gaze.origin.z - _ARM_M
            for track_id, face in faces.items()
            if face.in_3d is not None
        }
        bodies = {
            track_id: _place_hands(
                _lift_body(body, depth, intrinsics), depth, intrinsics, nearest.get(track_id)
            )
            for track_id, body in scene.bodies.items()
            if track_id != main_user
        }
        if main_user is not None and (body := scene.bodies.get(main_user)) is not None:
            sizes = self._sizes.setdefault(main_user, deque())
            while sizes and time - sizes[0][0] > _SIZES_KEPT_S:
                sizes.popleft()
            size = statistics.median(size for _, size in sizes) if sizes else 1.0
            previous = {
                side: None if (kept := self._hands.get((main_user, side))) is None else kept[0]
                for side in _SIDES
            }
            lifted = _lift_body(body, depth, intrinsics)
            bodies[main_user], fits = _lift_hands(
                lifted, depth, intrinsics, size, previous, nearest.get(main_user)
            )
            self._hands |= {(main_user, side): (params, time) for side, params in fits.items()}
            sizes.extend((time, math.exp(params[3])) for params in fits.values())
        return scene.model_copy(update={"faces": faces, "bodies": bodies})


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


def _lift_body(body: Body, depth: DepthMap, intrinsics: CameraIntrinsics) -> Body:
    """The body's keypoints placed in the camera's frame, each at the depth around it."""
    return body.model_copy(
        update={
            "keypoints": tuple(
                None
                if k is None
                else k.model_copy(update={"position": back_project(k.x, k.y, depth, intrinsics)})
                for k in body.keypoints
            )
        }
    )


def _lift_hands(
    body: Body,
    depth: DepthMap,
    intrinsics: CameraIntrinsics,
    size: float,
    previous: Mapping[Side, Params | None],
    nearest: float | None,
) -> tuple[Body, dict[Side, Params]]:
    """The body with its hands placed in the camera's frame, and the fit of each.

    Each hand's wrist is at the depth of its arm's wrist, which the forearm holds, so that the two
    skeletons meet; at its palm's depth when the arm's wrist was not found.

    Args:
        body: The skeleton, its keypoints placed, its hands in pixels of the frame.
        depth: The frame's depth.
        intrinsics: The camera at the frame's resolution.
        size: The size against the average hand the hands are held near.
        previous: The last fit of each of the person's hands, if any.
        nearest: The nearest a hand's wrist may be, from the person's face; `None` without it.
    """
    lifted: dict[Side, Hand | None] = {}
    fits: dict[Side, Params] = {}
    for side, hand, arm in zip(_SIDES, (body.left_hand, body.right_hand), _ARM_WRISTS, strict=True):
        wrist = None if hand is None else hand.keypoints[0]
        if hand is None or wrist is None:
            lifted[side] = hand
            continue
        z = _hand_depth(hand, _position(body.keypoints, arm), depth, intrinsics, nearest)
        found = {index: k for index, k in enumerate(hand.keypoints) if k is not None}
        at = _unproject(np.array([[wrist.x, wrist.y]]), np.array([z]), intrinsics)[0]
        points, fits[side] = fit(found, at, side, intrinsics, size, previous[side])
        lifted[side] = Hand(
            keypoints=tuple(
                None
                if k is None
                else k.model_copy(
                    update={
                        "position": Point3D(
                            x=float(points[index][0]),
                            y=float(points[index][1]),
                            z=float(points[index][2]),
                        )
                    }
                )
                for index, k in enumerate(hand.keypoints)
            )
        )
    return body.model_copy(
        update={"left_hand": lifted["left"], "right_hand": lifted["right"]}
    ), fits


def _place_hands(
    body: Body, depth: DepthMap, intrinsics: CameraIntrinsics, nearest: float | None
) -> Body:
    """The body with each hand's points placed at the depth of its wrist, as on a flat hand.

    Cheaper than a fit, and good enough to tell where the hand is: its depth around the fingers
    is often the wall's behind. `nearest` is the nearest a wrist may be, from the person's face;
    `None` without it.
    """
    placed: dict[Side, Hand | None] = {}
    for side, hand, arm in zip(_SIDES, (body.left_hand, body.right_hand), _ARM_WRISTS, strict=True):
        if hand is None or not (found := [k for k in hand.keypoints if k is not None]):
            placed[side] = hand
            continue
        wrist_z = _hand_depth(hand, _position(body.keypoints, arm), depth, intrinsics, nearest)
        pixels = np.array([(k.x, k.y) for k in found])
        points = _unproject(pixels, np.full(len(found), wrist_z), intrinsics).tolist()
        positions = iter(Point3D(x=x, y=y, z=z) for x, y, z in points)
        placed[side] = Hand(
            keypoints=tuple(
                None if k is None else k.model_copy(update={"position": next(positions)})
                for k in hand.keypoints
            )
        )
    return body.model_copy(update={"left_hand": placed["left"], "right_hand": placed["right"]})


def _hand_depth(
    hand: Hand,
    arm_wrist: Point3D | None,
    depth: DepthMap,
    intrinsics: CameraIntrinsics,
    nearest: float | None,
) -> float:
    """How far the hand's wrist is: its arm's wrist's depth, or else its palm's.

    No nearer than `nearest`, when known: the depth model reads a raised arm too near.
    """
    if arm_wrist is not None:
        z = arm_wrist.z
    else:
        found = [k for k in hand.keypoints if k is not None]
        on_palm = [k for index, k in enumerate(hand.keypoints) if k is not None and index in _PALM]
        # A hand seen without its palm, e.g. only fingertips, is placed at all its points' depth.
        pixels = np.array([(k.x, k.y) for k in on_palm or found])
        z = float(np.median(_depths(pixels, depth, intrinsics)))
    return z if nearest is None else max(z, nearest)


def _position(keypoints: tuple[Keypoint | None, ...], index: int) -> Point3D | None:
    keypoint = keypoints[index] if index < len(keypoints) else None
    return None if keypoint is None else keypoint.position
