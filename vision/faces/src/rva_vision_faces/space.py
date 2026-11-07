import math
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

_PATCH_RADIUS: Final = 2
"""Half the side of the square of depth pixels a point's depth is read from."""
_NEAR_PERCENTILE: Final = 25
"""Keypoints sit on the edges of limbs and fingers, where the patch also holds what is behind
them: the nearer depths are the person's."""
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""


def lift(scene: Scene, depth: DepthMap, intrinsics: CameraIntrinsics) -> Scene:
    """The scene with its keypoints and gazes placed in the camera's frame.

    Args:
        scene: What was seen, in pixels of the frame.
        depth: The frame's depth.
        intrinsics: The camera at the frame's resolution.
    """
    return scene.model_copy(
        update={
            "faces": {
                track_id: _lift_face(face, depth, intrinsics)
                for track_id, face in scene.faces.items()
            },
            "bodies": {
                track_id: _lift_body(body, depth, intrinsics)
                for track_id, body in scene.bodies.items()
            },
            "unmatched_bodies": tuple(
                _lift_body(body, depth, intrinsics) for body in scene.unmatched_bodies
            ),
        }
    )


def back_project(x: float, y: float, depth: DepthMap, intrinsics: CameraIntrinsics) -> Point3D:
    """The point seen at pixel (`x`, `y`) of the frame, at the depth around it."""
    ((px, py, pz),) = _back_project_all(np.array([[x, y]]), depth, intrinsics)
    return Point3D(x=float(px), y=float(py), z=float(pz))


def _back_project_all(
    pixels: npt.NDArray[np.float64], depth: DepthMap, intrinsics: CameraIntrinsics
) -> npt.NDArray[np.float64]:
    """`back_project` for N pixels at once, N x 2 in, N x 3 out; patches stop at the border."""
    rows, columns = depth.metres.shape
    offsets = np.arange(-_PATCH_RADIUS, _PATCH_RADIUS + 1)
    row = np.rint(pixels[:, 1] * rows / intrinsics.height).astype(np.intp)
    column = np.rint(pixels[:, 0] * columns / intrinsics.width).astype(np.intp)
    patch_rows = np.clip(row[:, None, None] + offsets[None, :, None], 0, rows - 1)
    patch_columns = np.clip(column[:, None, None] + offsets[None, None, :], 0, columns - 1)
    patches = depth.metres[patch_rows, patch_columns].reshape(len(pixels), -1)
    z = np.percentile(patches, _NEAR_PERCENTILE, axis=1).astype(np.float64)
    return np.column_stack(
        [
            (pixels[:, 0] - intrinsics.cx) / intrinsics.fx * z,
            (pixels[:, 1] - intrinsics.cy) / intrinsics.fy * z,
            z,
        ]
    )


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


def _lift_face(face: FaceFeatures, depth: DepthMap, intrinsics: CameraIntrinsics) -> FaceFeatures:
    (left_x, left_y), (right_x, right_y) = (face.landmarks[iris] for iris in _IRISES)
    eyes = back_project((left_x + right_x) / 2, (left_y + right_y) / 2, depth, intrinsics)
    landmarks = _back_project_all(np.array(face.landmarks), depth, intrinsics)
    in_3d = Face3D(
        landmarks=tuple(Point3D(x=x, y=y, z=z) for x, y, z in landmarks.tolist()),
        gaze=GazeRay(origin=eyes, direction=gaze_direction(eyes, face.gaze)),
    )
    return face.model_copy(update={"in_3d": in_3d})


def _lift_body(body: Body, depth: DepthMap, intrinsics: CameraIntrinsics) -> Body:
    def points(keypoints: tuple[Keypoint | None, ...]) -> tuple[Keypoint | None, ...]:
        return tuple(
            None
            if k is None
            else k.model_copy(update={"position": back_project(k.x, k.y, depth, intrinsics)})
            for k in keypoints
        )

    def hand(found: Hand | None) -> Hand | None:
        return None if found is None else Hand(keypoints=points(found.keypoints))

    return body.model_copy(
        update={
            "keypoints": points(body.keypoints),
            "left_hand": hand(body.left_hand),
            "right_hand": hand(body.right_hand),
        }
    )
