import math

import cv2
import numpy as np
import pytest

from rva_schemas.vision import Body, BoundingBox, Gaze, Hand, Keypoint
from rva_vision_faces.body import attach, decode_simcc, decode_yolox, pose_transform
from rva_vision_faces.detector import Detection
from rva_vision_faces.face_features import box_transform, gaze_from_bins, mesh_transform


def _body(nose: tuple[float, float] | None) -> Body:
    head = None if nose is None else Keypoint(x=nose[0], y=nose[1], score=0.9)
    return Body(
        box=BoundingBox(x=0, y=0, width=100, height=300),
        keypoints=(head, head, head, *[None] * 20),
        left_hand=Hand(keypoints=(None,) * 21),
        right_hand=Hand(keypoints=(None,) * 21),
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
