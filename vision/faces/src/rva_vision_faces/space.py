import math
from typing import Final

import numpy as np

from rva_schemas.vision import (
    Body,
    CameraIntrinsics,
    DepthMap,
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
    rows, columns = depth.metres.shape
    column = min(max(round(x * columns / intrinsics.width), 0), columns - 1)
    row = min(max(round(y * rows / intrinsics.height), 0), rows - 1)
    patch = depth.metres[
        max(row - _PATCH_RADIUS, 0) : row + _PATCH_RADIUS + 1,
        max(column - _PATCH_RADIUS, 0) : column + _PATCH_RADIUS + 1,
    ]
    z = float(np.percentile(patch, _NEAR_PERCENTILE))
    return Point3D(
        x=(x - intrinsics.cx) / intrinsics.fx * z,
        y=(y - intrinsics.cy) / intrinsics.fy * z,
        z=z,
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
    ray = GazeRay(origin=eyes, direction=gaze_direction(eyes, face.gaze))
    return face.model_copy(update={"gaze_ray": ray})


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
