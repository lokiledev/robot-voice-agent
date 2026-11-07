import math

import numpy as np
import pytest

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
    hand = Hand(keypoints=(Keypoint(x=1460, y=790, score=0.9, position=None), None))
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


def test_a_thumb_tip_with_nothing_found_up_the_hand_stays_near_the_palm() -> None:
    depth = _depth(3.0)
    # The index finger's base, 0.5 m away, is over the hand; the thumb's tip, with neither its
    # joints nor the wrist found, is over the wall.
    depth.metres[:, :240] = 0.5
    hand = _hand({5: (500, 540), 4: (1460, 540)})

    lifted = Lifter(INTRINSICS).lift(_scene(_body({}, hand)), depth)

    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None
    base, tip = left_hand.keypoints[5], left_hand.keypoints[4]
    assert base is not None and base.position is not None
    assert tip is not None and tip.position is not None
    assert base.position.z == pytest.approx(0.5)
    assert _xyz(tip.position) == pytest.approx((0.3, 0.0, 0.6), abs=1e-6)


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


def test_a_straight_finger_in_front_of_the_wall_keeps_its_length() -> None:
    # The index finger points right, 0.5 m away: the wrist at x = 0, its base 9 cm on, then
    # 4.2, 2.5 and 2.2 cm. Only the wrist and the base are over the hand in the depth; the
    # rest of the finger is too thin for it, which sees the wall 3 m behind.
    depth = _depth(3.0)
    depth.metres[:, :290] = 0.5
    hand = _hand({0: (960, 540), 5: (1140, 540), 6: (1224, 540), 7: (1274, 540), 8: (1318, 540)})

    lifted = Lifter(INTRINSICS).lift(_scene(_body({}, hand)), depth)

    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None
    placed = [k.position for k in left_hand.keypoints if k is not None]
    assert [_xyz(p) for p in placed if p is not None] == [
        pytest.approx((x, 0.0, 0.5), abs=1e-6) for x in (0.0, 0.09, 0.132, 0.157, 0.179)
    ]


def test_hands_are_scaled_with_the_shoulders() -> None:
    # Shoulders 45.6 cm apart, 2 m away: 1.2 times the standard person's, the most allowed.
    shoulders = {5: (1074.0, 300.0), 6: (846.0, 300.0)}
    hand = _hand({0: (960, 540), 5: (1010, 540)})

    lifted = Lifter(INTRINSICS).lift(_scene(_body(shoulders, hand)), _depth(2.0))

    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None
    wrist, base = left_hand.keypoints[0], left_hand.keypoints[5]
    assert wrist is not None and wrist.position is not None
    assert base is not None and base.position is not None
    assert math.dist(_xyz(wrist.position), _xyz(base.position)) == pytest.approx(0.09 * 1.2)


def test_a_tip_whose_joints_were_not_found_is_reached_from_the_base() -> None:
    # The straight finger again, with only its base and tip found: the tip is 6.9 cm on, the
    # three bones' summed length.
    depth = _depth(3.0)
    depth.metres[:, :290] = 0.5
    hand = _hand({0: (960, 540), 5: (1140, 540), 8: (1318, 540)})

    lifted = Lifter(INTRINSICS).lift(_scene(_body({}, hand)), depth)

    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None
    tip = left_hand.keypoints[8]
    assert tip is not None and tip.position is not None
    assert _xyz(tip.position) == pytest.approx((0.179, 0.0, 0.5), abs=1e-6)


def test_a_wrist_over_the_wall_is_brought_as_near_as_the_hand_looks() -> None:
    # The hand of the straight finger, 0.5 m away, whose wrist the depth model sees through:
    # 9 cm between the wrist and the index finger's base look 180 pixels long, which a hand
    # farther than 0.5 m could not.
    depth = _depth(3.0)
    depth.metres[:, 250:290] = 0.5
    hand = _hand({0: (960, 540), 5: (1140, 540)})

    lifted = Lifter(INTRINSICS).lift(_scene(_body({}, hand)), depth)

    left_hand = lifted.bodies[1].left_hand
    assert left_hand is not None
    wrist, base = left_hand.keypoints[0], left_hand.keypoints[5]
    assert wrist is not None and wrist.position is not None
    assert base is not None and base.position is not None
    assert _xyz(wrist.position) == pytest.approx((0.0, 0.0, 0.5), abs=1e-6)
    assert _xyz(base.position) == pytest.approx((0.09, 0.0, 0.5), abs=1e-6)
