import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rva_schemas.vision import (
    Body,
    BoundingBox,
    CameraIntrinsics,
    DepthMap,
    FaceFeatures,
    Gaze,
    Hand,
    Keypoint,
    Point3D,
)
from rva_vision_faces.hand_model import PARAMS, hand_points
from rva_vision_faces.scene import Scene
from rva_vision_faces.space import Lifter, back_project, gaze_direction

INTRINSICS = CameraIntrinsics(
    calibrated=False, width=1920, height=1080, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
)


def _depth(metres: float) -> DepthMap:
    """A flat depth at a quarter of the frame's resolution."""
    return DepthMap(metres=np.full((270, 480), metres, np.float32), calibrated=False)


def _xyz(point: Point3D) -> tuple[float, float, float]:
    return point.x, point.y, point.z


def test_a_pixel_is_placed_at_its_depth_along_its_ray() -> None:
    assert _xyz(back_project(960, 540, _depth(2.0), INTRINSICS)) == pytest.approx(
        (0.0, 0.0, 2.0), abs=1e-6
    )
    # 500 pixels right and 250 down of the center, with a 1000-pixel focal length.
    assert _xyz(back_project(1460, 790, _depth(2.0), INTRINSICS)) == pytest.approx(
        (1.0, 0.5, 2.0), abs=1e-6
    )


def test_a_point_on_an_edge_takes_the_nearer_side() -> None:
    depth = _depth(3.0)
    # A forearm 1 m away fills two fifths of the patch, the wall behind the rest.
    depth.metres[:, :240] = 1.0

    assert back_project(962, 540, depth, INTRINSICS).z == pytest.approx(1.0)


def test_points_out_of_the_frame_read_the_depth_at_its_edge() -> None:
    depth = _depth(2.0)
    depth.metres[:, -3:] = 4.0

    assert back_project(1950, 540, depth, INTRINSICS).z == pytest.approx(4.0)


def test_a_gaze_of_zero_looks_back_at_the_camera() -> None:
    eyes = Point3D(x=0.6, y=-0.2, z=2.0)

    direction = gaze_direction(eyes, Gaze(yaw=0.0, pitch=0.0))

    distance = math.hypot(0.6, -0.2, 2.0)
    assert _xyz(direction) == pytest.approx(
        (-0.6 / distance, 0.2 / distance, -2.0 / distance), abs=1e-6
    )


def test_yaw_turns_the_gaze_towards_the_right_of_the_image_and_pitch_upwards() -> None:
    eyes = Point3D(x=0.0, y=0.0, z=2.0)

    assert _xyz(gaze_direction(eyes, Gaze(yaw=90.0, pitch=0.0))) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-6
    )
    assert _xyz(gaze_direction(eyes, Gaze(yaw=0.0, pitch=90.0))) == pytest.approx(
        (0.0, -1.0, 0.0), abs=1e-6
    )
    half = math.sqrt(0.5)
    assert _xyz(gaze_direction(eyes, Gaze(yaw=45.0, pitch=0.0))) == pytest.approx(
        (half, 0.0, -half), abs=1e-6
    )


def test_lifting_places_keypoints_and_gazes_and_leaves_the_rest() -> None:
    hand = _hand({0: (1460, 790)})
    body = Body(
        box=BoundingBox(x=500, y=300, width=900, height=780),
        keypoints=(Keypoint(x=960, y=540, score=0.8, position=None), None),
        left_hand=hand,
        right_hand=None,
    )
    landmarks = [(0.0, 0.0)] * 478
    # The irises, either side of the center.
    landmarks[468], landmarks[473] = (900.0, 540.0), (1020.0, 540.0)
    face = FaceFeatures(
        landmarks=tuple(landmarks),
        left_eye_open=1.0,
        right_eye_open=1.0,
        mouth_open=0.0,
        gaze=Gaze(yaw=0.0, pitch=0.0),
        in_3d=None,
        expression="neutral",
        expression_score=0.9,
    )
    scene = Scene(faces={1: face}, bodies={1: body}, unmatched_bodies=(body,), speakers={})

    lifted = Lifter(INTRINSICS).lift(scene, _depth(2.0))

    nose = lifted.bodies[1].keypoints[0]
    assert nose is not None and nose.position is not None
    assert _xyz(nose.position) == pytest.approx((0.0, 0.0, 2.0), abs=1e-6)
    assert lifted.bodies[1].keypoints[1] is None
    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None and left_hand.keypoints[0] is not None
    assert left_hand.keypoints[0].position == Point3D(x=1.0, y=0.5, z=2.0)
    assert lifted.unmatched_bodies[0] == lifted.bodies[1]
    face_3d = lifted.faces[1].in_3d
    assert face_3d is not None
    assert _xyz(face_3d.landmarks[473]) == pytest.approx((0.12, 0.0, 2.0), abs=1e-6)
    assert _xyz(face_3d.landmarks[0]) == pytest.approx((-1.92, -1.08, 2.0), abs=1e-6)
    assert _xyz(face_3d.gaze.origin) == pytest.approx((0.0, 0.0, 2.0), abs=1e-6)
    assert _xyz(face_3d.gaze.direction) == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def _scene(body: Body) -> Scene:
    return Scene(faces={}, bodies={1: body}, unmatched_bodies=(), speakers={})


def _hand(points: dict[int, tuple[float, float]]) -> Hand:
    return Hand(
        keypoints=tuple(
            None
            if (pixel := points.get(index)) is None
            else Keypoint(x=pixel[0], y=pixel[1], score=0.9, position=None)
            for index in range(21)
        )
    )


def _body(keypoints: dict[int, tuple[float, float]], hand: Hand) -> Body:
    return Body(
        box=BoundingBox(x=0, y=0, width=1920, height=1080),
        keypoints=tuple(
            None
            if (pixel := keypoints.get(index)) is None
            else Keypoint(x=pixel[0], y=pixel[1], score=0.9, position=None)
            for index in range(13)
        ),
        left_hand=hand,
        right_hand=None,
    )


def test_a_hand_is_fitted_with_its_wrist_at_its_arm_s_wrist_depth() -> None:
    # A right hand, palm to the camera, its fingers a little curled, the wrist 0.6 m away. The
    # depth model sees the forearm, which ends at the arm's wrist, and the wall through the rest.
    truth = np.zeros(PARAMS)
    truth[:3] = Rotation.from_euler("x", math.pi).as_rotvec()
    truth[4:] = np.tile([0.3, 0.0, 0.3, 0.2], 5)
    wrist = np.array([0.05, 0.1, 0.6])
    points = wrist + hand_points(truth, "right")
    pixels = {
        index: (1000 * x / z + 960, 1000 * y / z + 540)
        for index, (x, y, z) in enumerate(points.tolist())
    }
    depth = _depth(3.0)
    depth.metres[:, 255:262] = 0.6
    arm = {10: (1043.0, 707.0)}
    body = Body(
        box=BoundingBox(x=0, y=0, width=1920, height=1080),
        keypoints=tuple(
            Keypoint(x=arm[index][0], y=arm[index][1], score=0.9, position=None)
            if index in arm
            else None
            for index in range(13)
        ),
        left_hand=None,
        right_hand=_hand(pixels),
    )

    lifted = Lifter(INTRINSICS).lift(_scene(body), depth)

    right_hand = lifted.bodies[1].right_hand
    assert right_hand is not None
    placed = np.array(
        [_xyz(k.position) for k in right_hand.keypoints if k is not None and k.position is not None]
    )
    assert np.abs(placed - points).max() == pytest.approx(0.0, abs=1e-3)
