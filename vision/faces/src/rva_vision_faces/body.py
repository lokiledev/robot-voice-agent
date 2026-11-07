from collections.abc import Sequence
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from rva_schemas.vision import BODY_CONNECTIONS, Body, BoundingBox, Hand, Keypoint
from rva_vision_faces.config import BodyConfig
from rva_vision_faces.detector import Image
from rva_vision_faces.tracker import iou

_DETECTOR_SIZE: Final = 640
_DETECTOR_STRIDES: Final = (8, 16, 32)
_PAD_VALUE: Final = 114
_PERSON: Final = 5
"""Column of the person class in YOLOX's outputs, after the box and the objectness."""
_POSE_WIDTH: Final = 192
_POSE_HEIGHT: Final = 256
_POSE_PADDING: Final = 1.25
"""The pose crop is this much larger than the person's box, as RTMW was trained."""
_SIMCC_SPLIT: Final = 2.0
"""RTMW locates keypoints on a grid twice as fine as its input."""
_POSE_MEAN: Final = np.array([123.675, 116.28, 103.53], np.float32)
_POSE_STD: Final = np.array([58.395, 57.12, 57.375], np.float32)
# Ranges of COCO-WholeBody's 133 keypoints: the upper body and the hands. The legs and feet are
# left out, and the 68 points of the face are left to the face mesh.
_BODY: Final = slice(0, 13)
_LEFT_HAND: Final = slice(91, 112)
_RIGHT_HAND: Final = slice(112, 133)
_SAME_HAND_IOU: Final = 0.5
"""Overlap of a skeleton's two hands above which they are one hand, found as both."""
_TORSO: Final = (0, 1, 2, 3, 4, 5, 6, 11, 12)
"""Head, shoulders and hips: the body keypoints the others must be linked to."""
_HEAD: Final = (0, 1, 2)
"""Nose and eyes: the keypoints that tie a skeleton to a face."""


class BodyEstimator:
    """Skeletons of the people in a frame: YOLOX finds them, RTMW places 133 keypoints on each."""

    def __init__(
        self,
        detector: onnxruntime.InferenceSession,
        pose: onnxruntime.InferenceSession,
        config: BodyConfig,
    ) -> None:
        self._detector: Final = detector
        self._pose: Final = pose
        self._config: Final = config

    def estimate(self, frame: Image) -> list[Body]:
        boxes = self._detect(frame)
        if not boxes:
            return []
        rgb = frame[:, :, ::-1]
        transforms = [pose_transform(box) for box in boxes]
        crops = np.stack([cv2.warpAffine(rgb, t, (_POSE_WIDTH, _POSE_HEIGHT)) for t in transforms])
        blob = ((crops - _POSE_MEAN) / _POSE_STD).transpose(0, 3, 1, 2).astype(np.float32)
        simcc_x, simcc_y = self._pose.run(None, {self._pose.get_inputs()[0].name: blob})
        keypoints = decode_simcc(np.asarray(simcc_x, np.float32), np.asarray(simcc_y, np.float32))
        return [
            _body(box, keypoints[index], transform, self._config.keypoint_threshold)
            for index, (box, transform) in enumerate(zip(boxes, transforms, strict=True))
        ]

    def _detect(self, frame: Image) -> list[BoundingBox]:
        height, width = frame.shape[:2]
        ratio = min(_DETECTOR_SIZE / height, _DETECTOR_SIZE / width)
        resized = cv2.resize(frame, (round(width * ratio), round(height * ratio)))
        padded = np.full((_DETECTOR_SIZE, _DETECTOR_SIZE, 3), _PAD_VALUE, np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        (output,) = self._detector.run(
            None, {self._detector.get_inputs()[0].name: padded.transpose(2, 0, 1)[None]}
        )
        return decode_yolox(
            np.asarray(output, np.float32)[0],
            ratio,
            self._config.score_threshold,
            self._config.nms_threshold,
        )


def decode_yolox(
    output: npt.NDArray[np.float32], ratio: float, score_threshold: float, nms_threshold: float
) -> list[BoundingBox]:
    """People's boxes from YOLOX's raw per-anchor outputs, after non-maximum suppression.

    Args:
        output: `anchors x 85`: center offset and log size in strides, objectness, 80 classes;
            anchors in row-major order over a 640x640 input, stride 8 first.
        ratio: Scale of the frame in the input.
        score_threshold: Minimum person score.
        nms_threshold: IoU above which the weaker of two detections is dropped.
    """
    grids, strides = [], []
    for stride in _DETECTOR_STRIDES:
        cells = _DETECTOR_SIZE // stride
        ys, xs = np.mgrid[:cells, :cells]
        grids.append(np.stack([xs.ravel(), ys.ravel()], axis=1))
        strides.append(np.full((cells * cells, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    scores = output[:, 4] * output[:, _PERSON]
    kept = scores >= score_threshold
    center = (output[kept, :2] + grid[kept]) * stride[kept] / ratio
    size = np.exp(output[kept, 2:4]) * stride[kept] / ratio
    corners = center - size / 2
    boxes = np.concatenate([corners, size], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores[kept].tolist(), score_threshold, nms_threshold
    )
    return [
        BoundingBox(
            x=float(boxes[i, 0]),
            y=float(boxes[i, 1]),
            width=float(boxes[i, 2]),
            height=float(boxes[i, 3]),
        )
        for i in np.asarray(indices, dtype=np.int64).reshape(-1)
    ]


def pose_transform(box: BoundingBox) -> npt.NDArray[np.float64]:
    """Affine map from the frame to RTMW's 192x256 crop, padded and widened to its aspect."""
    width, height = box.width * _POSE_PADDING, box.height * _POSE_PADDING
    aspect = _POSE_WIDTH / _POSE_HEIGHT
    if width > height * aspect:
        height = width / aspect
    else:
        width = height * aspect
    scale = _POSE_WIDTH / width
    center_x, center_y = box.x + box.width / 2, box.y + box.height / 2
    return np.array(
        [
            [scale, 0, _POSE_WIDTH / 2 - center_x * scale],
            [0, scale, _POSE_HEIGHT / 2 - center_y * scale],
        ]
    )


def decode_simcc(
    simcc_x: npt.NDArray[np.float32], simcc_y: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Keypoints in the crop from RTMW's per-axis distributions.

    Returns:
        `people x keypoints x (x, y, score)`, the score being the lower of the two peaks.
    """
    x = np.argmax(simcc_x, axis=2) / _SIMCC_SPLIT
    y = np.argmax(simcc_y, axis=2) / _SIMCC_SPLIT
    score = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
    return np.stack([x, y, score], axis=2).astype(np.float32)


def attach(
    bodies: Sequence[Body], faces: Sequence[tuple[int, BoundingBox]]
) -> tuple[dict[int, Body], list[Body]]:
    """Give each face the skeleton whose nose and eyes fall in its box, one skeleton per face.

    Args:
        bodies: Skeletons found in a frame.
        faces: Track ids and boxes of the faces seen in the same frame.

    Returns:
        The skeleton of each face that has one, by track id, and the skeletons left over.
    """
    pairs = sorted(
        (
            (hits, index, track_id)
            for index, body in enumerate(bodies)
            for track_id, box in faces
            if (hits := _head_points_in(body, box)) > 0
        ),
        reverse=True,
    )
    attached: dict[int, Body] = {}
    taken: set[int] = set()
    for _, index, track_id in pairs:
        if index not in taken and track_id not in attached:
            attached[track_id] = bodies[index]
            taken.add(index)
    return attached, [body for index, body in enumerate(bodies) if index not in taken]


def _head_points_in(body: Body, box: BoundingBox) -> int:
    return sum(
        1
        for index in _HEAD
        if (point := body.keypoints[index]) is not None
        and box.x <= point.x <= box.x + box.width
        and box.y <= point.y <= box.y + box.height
    )


def _body(
    box: BoundingBox,
    points: npt.NDArray[np.float32],
    transform: npt.NDArray[np.float64],
    threshold: float,
) -> Body:
    in_frame = cv2.transform(points[None, :, :2], cv2.invertAffineTransform(transform))[0]

    def keypoints(part: slice) -> tuple[Keypoint | None, ...]:
        return tuple(
            Keypoint(x=float(x), y=float(y), score=float(score)) if score >= threshold else None
            for (x, y), score in zip(in_frame[part], points[part, 2], strict=True)
        )

    left, right = one_hand_each(_hand(keypoints(_LEFT_HAND)), _hand(keypoints(_RIGHT_HAND)))
    return Body(
        box=box, keypoints=linked_to_torso(keypoints(_BODY)), left_hand=left, right_hand=right
    )


def linked_to_torso(keypoints: tuple[Keypoint | None, ...]) -> tuple[Keypoint | None, ...]:
    """Drop the body keypoints that no chain of found bones links to the head, shoulders or hips.

    RTMW can place a joint that is out of frame or hidden, such as a wrist whose elbow it did not
    find; with the joints between it and the body missing, it floats free.
    """
    linked = {index for index in _TORSO if keypoints[index] is not None}
    frontier = list(linked)
    while frontier:
        joint = frontier.pop()
        for bone in BODY_CONNECTIONS:
            if joint in bone:
                other = bone[1] if bone[0] == joint else bone[0]
                if other not in linked and keypoints[other] is not None:
                    linked.add(other)
                    frontier.append(other)
    return tuple(point if index in linked else None for index, point in enumerate(keypoints))


def one_hand_each(left: Hand | None, right: Hand | None) -> tuple[Hand | None, Hand | None]:
    """Keep the better of a skeleton's two hands when they cover the same hand.

    Without the arms in view, as when a hand is held up close, RTMW can put both of a person's
    hands on the one it sees. The weaker of the two, often mirrored, is dropped.
    """
    if left is None or right is None or iou(_extent(left), _extent(right)) < _SAME_HAND_IOU:
        return left, right
    return (left, None) if _mean_score(left) >= _mean_score(right) else (None, right)


def _hand(keypoints: tuple[Keypoint | None, ...]) -> Hand | None:
    return Hand(keypoints=keypoints) if any(point is not None for point in keypoints) else None


def _found(hand: Hand) -> list[Keypoint]:
    return [point for point in hand.keypoints if point is not None]


def _extent(hand: Hand) -> BoundingBox:
    xs, ys = [point.x for point in _found(hand)], [point.y for point in _found(hand)]
    return BoundingBox(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))


def _mean_score(hand: Hand) -> float:
    return sum(point.score for point in _found(hand)) / len(_found(hand))
