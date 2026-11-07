import math

import cv2
import numpy as np
import pytest

from rva_schemas.vision import Body, BoundingBox, Gaze, Hand, Keypoint
from rva_vision_faces.body import (
    attach,
    decode_simcc,
    decode_yolox,
    linked_to_torso,
    linked_to_wrist,
    one_hand_each,
    pose_transform,
    to_body,
)
from rva_vision_faces.config import CameraIntrinsics
from rva_vision_faces.depth import letterbox, metres_per_unit
from rva_vision_faces.detector import Detection
from rva_vision_faces.face_features import box_transform, gaze_from_bins, mesh_transform


def _body(nose: tuple[float, float] | None) -> Body:
    head = None if nose is None else Keypoint(x=nose[0], y=nose[1], score=0.9)
    return Body(
        box=BoundingBox(x=0, y=0, width=100, height=300),
        keypoints=(head, head, head, *[None] * 10),
        left_hand=None,
        right_hand=None,
    )


def test_decode_yolox_keeps_confident_people_in_frame_pixels() -> None:
    output = np.zeros((8400, 85), np.float32)
    # Stride 8, row 2, column 3: anchor 2 * 80 + 3; a person, 32 by 64 input pixels.
    output[163, :4] = [0.5, 0.5, math.log(4), math.log(8)]
    output[163, 4:6] = [1.0, 0.9]
    # A confident detection of another class is not a person.
    output[200, 4:7] = [1.0, 0.0, 0.9]

    (box,) = decode_yolox(output, ratio=0.5, score_threshold=0.5, nms_threshold=0.45)

    # Center (3.5, 2.5) * 8 = (28, 20) in the input, (56, 40) in the frame; size 64 by 128.
    assert box == BoundingBox(x=24.0, y=-24.0, width=64.0, height=128.0)


def test_pose_transform_centers_the_box_padded_to_the_crop_aspect() -> None:
    # 100 x 400 is taller than 3:4: padded to 125 x 500, then widened to 375 x 500.
    transform = pose_transform(BoundingBox(x=50, y=0, width=100, height=400))

    corners = cv2.transform(np.array([[[100, 200], [100 - 187.5, -50]]], np.float32), transform)[0]
    assert corners == pytest.approx(np.array([[96, 128], [0, 0]]), abs=1e-4)


def test_decode_simcc_takes_each_axis_peak_at_half_a_pixel() -> None:
    simcc_x = np.zeros((1, 2, 384), np.float32)
    simcc_y = np.zeros((1, 2, 512), np.float32)
    simcc_x[0, 0, 101], simcc_y[0, 0, 40] = 0.9, 0.6
    simcc_x[0, 1, 0], simcc_y[0, 1, 511] = 0.2, 0.8

    assert decode_simcc(simcc_x, simcc_y) == pytest.approx(
        np.array([[[50.5, 20.0, 0.6], [0.0, 255.5, 0.2]]])
    )


def test_attach_gives_each_face_the_skeleton_whose_head_is_in_its_box() -> None:
    alice_face = BoundingBox(x=100, y=100, width=50, height=60)
    bob_face = BoundingBox(x=400, y=100, width=50, height=60)
    alice, far_away, turned_away = _body((125, 130)), _body((900, 50)), _body(None)

    attached, unmatched = attach([far_away, alice, turned_away], [(1, alice_face), (2, bob_face)])

    assert attached == {1: alice}
    assert unmatched == [far_away, turned_away]


def test_gaze_from_bins_reads_the_expected_angle_in_image_directions() -> None:
    # Bin 47 is +8 degrees: for L2CS-Net, towards the left of the image, and upwards.
    probabilities = np.zeros(90, np.float32)
    probabilities[47] = 1.0

    assert gaze_from_bins(probabilities, probabilities) == Gaze(yaw=-8.0, pitch=8.0)


def test_mesh_transform_levels_the_eyes_and_centers_the_face() -> None:
    # Eyes tilted by 45 degrees, around a face box centered on (200, 200).
    detection = Detection(
        box=BoundingBox(x=150, y=150, width=100, height=100),
        landmarks=((180, 180), (220, 220), (200, 200), (190, 230), (230, 190)),
        score=0.9,
    )

    points = cv2.transform(
        np.array([[[200, 200], [180, 180], [220, 220]]], np.float32), mesh_transform(detection)
    )[0]

    assert points[0] == pytest.approx([128, 128])
    assert points[1, 1] == pytest.approx(points[2, 1])
    # 150 pixels of frame across the 256 of the crop.
    assert points[2, 0] - points[1, 0] == pytest.approx(math.hypot(40, 40) * 256 / 150)


def test_box_transform_maps_the_box_onto_the_crop() -> None:
    transform = box_transform(BoundingBox(x=10, y=20, width=100, height=50), 200)

    corners = cv2.transform(np.array([[[10, 20], [110, 70]]], np.float32), transform)[0]
    assert corners == pytest.approx(np.array([[0, 0], [200, 200]]), abs=1e-4)


def _hand_at(x: float, score: float) -> Hand:
    """A hand spread over 100 by 100 pixels from `x`, its points all scoring `score`."""
    return Hand(
        keypoints=tuple(Keypoint(x=x + 5 * (i % 21), y=5.0 * i, score=score) for i in range(21))
    )


def test_one_hand_each_keeps_the_better_of_two_hands_on_the_same_hand() -> None:
    weak_left, strong_right = _hand_at(100, 0.4), _hand_at(105, 0.8)

    assert one_hand_each(weak_left, strong_right) == (None, strong_right)
    assert one_hand_each(_hand_at(100, 0.9), strong_right) == (_hand_at(100, 0.9), None)


def test_one_hand_each_keeps_both_hands_when_they_are_apart() -> None:
    left, right = _hand_at(100, 0.4), _hand_at(400, 0.8)

    assert one_hand_each(left, right) == (left, right)
    assert one_hand_each(None, right) == (None, right)


def test_linked_to_torso_drops_a_wrist_whose_elbow_was_not_found() -> None:
    def found(*indices: int) -> tuple[Keypoint | None, ...]:
        return tuple(
            Keypoint(x=float(i), y=0.0, score=0.5) if i in indices else None for i in range(13)
        )

    # The right arm is whole; the left wrist floats, its elbow out of frame.
    assert linked_to_torso(found(0, 1, 2, 5, 6, 8, 10, 9)) == found(0, 1, 2, 5, 6, 8, 10)
    # Without the head and shoulders, the hips anchor the torso; the arms hang off nothing.
    assert linked_to_torso(found(11, 12, 7, 9)) == found(11, 12)


def test_linked_to_wrist_drops_finger_segments_floating_on_their_own() -> None:
    def found(*indices: int) -> tuple[Keypoint | None, ...]:
        return tuple(
            Keypoint(x=float(i), y=0.0, score=0.5) if i in indices else None for i in range(21)
        )

    # Thumb and index whole; the ring finger's tip end without its base (13).
    assert linked_to_wrist(found(0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16)) == found(
        0, 1, 2, 3, 4, 5, 6, 7, 8
    )
    # No wrist: nothing holds the fingers to a hand.
    assert linked_to_wrist(found(1, 2, 3, 4)) == found()


def test_a_weak_double_of_a_hand_is_dropped_before_its_loose_points() -> None:
    points = np.zeros((133, 3), np.float32)
    # Right hand (112 to 132): 21 points spread over 100 by 100 pixels, all confident.
    points[112:133, 0] = 100 + 5 * np.arange(21)
    points[112:133, 1] = 5 * np.arange(21)
    points[112:133, 2] = 0.9
    # Left hand (91 to 111): the same places, weaker, without its wrist; filtered first, it
    # would shrink to nothing and no longer overlap.
    points[91:112] = points[112:133]
    points[91:112, 2] = 0.4
    points[91, 2] = 0.0

    body = to_body(
        BoundingBox(x=0, y=0, width=300, height=300),
        points,
        np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        threshold=0.3,
    )

    assert body.left_hand is None
    assert body.right_hand is not None


def test_letterbox_pads_a_landscape_frame_above_and_below() -> None:
    frame = np.full((1080, 1920, 3), 255, np.uint8)

    square, rows = letterbox(frame, 504)

    assert square.shape == (504, 504, 3)
    assert rows == slice(110, 394)
    assert (square[rows] == 255).all()
    assert square[0, 0].tolist() == [124, 116, 104]
    assert square[-1, 0].tolist() == [124, 116, 104]


def test_intrinsics_scale_with_the_resolution() -> None:
    intrinsics = CameraIntrinsics(width=1920, height=1080, fx=1400, fy=1380, cx=950, cy=530)

    scaled = intrinsics.scaled_to(1280, 720)

    assert (scaled.width, scaled.height) == (1280, 720)
    assert (scaled.fx, scaled.fy, scaled.cx, scaled.cy) == pytest.approx(
        (1400 * 2 / 3, 920, 950 * 2 / 3, 530 * 2 / 3)
    )
    with pytest.raises(AssertionError):
        intrinsics.scaled_to(640, 480)


def test_depth_is_scaled_by_the_focal_length_at_the_network_input() -> None:
    intrinsics = CameraIntrinsics(width=1920, height=1080, fx=1400, fy=1380, cx=960, cy=540)

    # A 1390 px mean focal at 1920 wide is 364.875 px at 504 wide; the network assumes 300.
    assert metres_per_unit(intrinsics, 504) == pytest.approx(364.875 / 300)
