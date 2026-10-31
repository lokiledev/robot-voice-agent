from collections.abc import Mapping
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from rva_pydantic import RVABaseModel
from rva_schemas.vision import BoundingBox
from rva_vision_faces.config import DetectorConfig

type Image = npt.NDArray[np.uint8]
"""A BGR image, height x width x 3, as OpenCV captures it."""

_STRIDES: Final = (8, 16, 32)
_DIVISOR: Final = 32


class Detection(RVABaseModel):
    box: BoundingBox
    landmarks: tuple[tuple[float, float], ...]
    """Right eye, left eye, nose tip, right and left mouth corners, in pixels."""
    score: float


class YunetDetector:
    """Face detection with YuNet (opencv_zoo, 2023mar) on frames of any size."""

    def __init__(self, session: onnxruntime.InferenceSession, config: DetectorConfig) -> None:
        self._session: Final = session
        self._config: Final = config
        self._input: Final = session.get_inputs()[0].name
        self._outputs: Final = [output.name for output in session.get_outputs()]

    def detect(self, frame: Image) -> list[Detection]:
        height, width = frame.shape[:2]
        # The network halves the size five times: pad on the bottom and right to keep coordinates.
        padded = np.zeros(
            (1, 3, _round_up(height, _DIVISOR), _round_up(width, _DIVISOR)), np.float32
        )
        padded[0, :, :height, :width] = frame.transpose(2, 0, 1)
        outputs = self._session.run(self._outputs, {self._input: padded})
        return decode_yunet(
            {
                name: np.asarray(output, np.float32)
                for name, output in zip(self._outputs, outputs, strict=True)
            },
            padded.shape[3],
            self._config.score_threshold,
            self._config.nms_threshold,
        )


def decode_yunet(
    outputs: Mapping[str, npt.NDArray[np.float32]],
    width: int,
    score_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    """Turn YuNet's per-anchor outputs into detections, after non-maximum suppression.

    Args:
        outputs: `cls_<stride>`, `obj_<stride>`, `bbox_<stride>` and `kps_<stride>`, each
            `1 x anchors x n` with anchors in row-major order over the padded input.
        width: Width of the padded input.
        score_threshold: Minimum detection score.
        nms_threshold: IoU above which the weaker of two detections is dropped.
    """
    boxes: list[npt.NDArray[np.float32]] = []
    scores: list[npt.NDArray[np.float32]] = []
    landmarks: list[npt.NDArray[np.float32]] = []
    for stride in _STRIDES:
        score = np.sqrt(
            np.clip(outputs[f"cls_{stride}"][0, :, 0], 0, 1)
            * np.clip(outputs[f"obj_{stride}"][0, :, 0], 0, 1)
        )
        anchors = np.nonzero(score >= score_threshold)[0]
        rows, cols = np.divmod(anchors, width // stride)
        box = outputs[f"bbox_{stride}"][0, anchors]
        size = np.exp(box[:, 2:4]) * stride
        center = (np.stack([cols, rows], axis=1) + box[:, 0:2]) * stride
        boxes.append(np.concatenate([center - size / 2, size], axis=1))
        scores.append(score[anchors])
        points = outputs[f"kps_{stride}"][0, anchors].reshape(-1, 5, 2)
        landmarks.append((points + np.stack([cols, rows], axis=1)[:, None, :]) * stride)
    all_boxes = np.concatenate(boxes)
    all_scores = np.concatenate(scores)
    all_landmarks = np.concatenate(landmarks)
    kept = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), score_threshold, nms_threshold)
    return [
        Detection(
            box=BoundingBox(
                x=float(all_boxes[i, 0]),
                y=float(all_boxes[i, 1]),
                width=float(all_boxes[i, 2]),
                height=float(all_boxes[i, 3]),
            ),
            landmarks=tuple((float(x), float(y)) for x, y in all_landmarks[i]),
            score=float(all_scores[i]),
        )
        for i in np.asarray(kept, dtype=np.int64).reshape(-1)
    ]


def _round_up(value: int, divisor: int) -> int:
    return -(-value // divisor) * divisor
