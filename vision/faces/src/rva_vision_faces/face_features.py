import math
from collections.abc import Sequence
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from rva_schemas.vision import BoundingBox, Expression, FaceFeatures, Gaze
from rva_vision_faces.detector import Detection, Image

_MESH_SIZE: Final = 256
_MESH_ROI_SCALE: Final = 1.5
"""The mesh crop is this much larger than the face box, as in MediaPipe's face landmarker."""
_GAZE_SIZE: Final = 448
_GAZE_BINS: Final = 4.0
"""L2CS-Net's 90 bins are 4 degrees wide, from -180."""
_EXPRESSION_SIZE: Final = 224
_IMAGENET_MEAN: Final = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD: Final = np.array([0.229, 0.224, 0.225], np.float32)
_EXPRESSIONS: Final[tuple[Expression, ...]] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
"""FER2013's labels, in the order of the model's outputs."""
# Mesh points the blendshape model reads (MediaPipe's face_blendshapes_graph.cc).
_BLENDSHAPE_POINTS: Final = np.array(
    [
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
        65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
        136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
        168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270,
        276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
        318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379,
        380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
        466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ]
)  # fmt: skip
_EYE_BLINK_LEFT: Final = 9
_EYE_BLINK_RIGHT: Final = 10
_JAW_OPEN: Final = 25


class FaceAnalyzer:
    """Eyes, gaze, mouth and expression of detected faces, each from a model of its own.

    MediaPipe's face mesh gives the points, its blendshapes how open the eyes and the mouth are,
    L2CS-Net the gaze, and a ViT fine-tuned on FER2013 the expression.
    """

    def __init__(
        self,
        mesh: onnxruntime.InferenceSession,
        blendshapes: onnxruntime.InferenceSession,
        gaze: onnxruntime.InferenceSession,
        expression: onnxruntime.InferenceSession,
    ) -> None:
        self._mesh: Final = mesh
        self._blendshapes: Final = blendshapes
        self._gaze: Final = gaze
        self._expression: Final = expression

    def analyze(self, frame: Image, detections: Sequence[Detection]) -> list[FaceFeatures]:
        """The features of each face, in the order of `detections`."""
        if not detections:
            return []
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        landmarks = self._landmarks(rgb, detections)
        expressions = self._expressions(frame, detections)
        features = []
        for index, detection in enumerate(detections):
            blendshapes = self._run_blendshapes(landmarks[index])
            best = int(np.argmax(expressions[index]))
            features.append(
                FaceFeatures(
                    landmarks=tuple((float(x), float(y)) for x, y in landmarks[index]),
                    left_eye_open=1.0 - float(blendshapes[_EYE_BLINK_LEFT]),
                    right_eye_open=1.0 - float(blendshapes[_EYE_BLINK_RIGHT]),
                    mouth_open=float(blendshapes[_JAW_OPEN]),
                    gaze=self._run_gaze(rgb, detection.box),
                    expression=_EXPRESSIONS[best],
                    expression_score=float(expressions[index, best]),
                )
            )
        return features

    def _landmarks(
        self, rgb: Image, detections: Sequence[Detection]
    ) -> list[npt.NDArray[np.float32]]:
        transforms = [mesh_transform(detection) for detection in detections]
        crops = np.stack(
            [cv2.warpAffine(rgb, transform, (_MESH_SIZE, _MESH_SIZE)) for transform in transforms]
        )
        outputs = self._mesh.run(None, {self._mesh.get_inputs()[0].name: crops / np.float32(255)})
        points = np.asarray(outputs[0], np.float32).reshape(len(detections), -1, 3)[:, :, :2]
        return [
            cv2.transform(crop_points[None], cv2.invertAffineTransform(transform))[0]
            for crop_points, transform in zip(points, transforms, strict=True)
        ]

    def _run_blendshapes(self, landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # MediaPipe feeds the points in image pixels.
        points = landmarks[_BLENDSHAPE_POINTS][None].astype(np.float32)
        (scores,) = self._blendshapes.run(None, {self._blendshapes.get_inputs()[0].name: points})
        return np.asarray(scores, np.float32)

    def _run_gaze(self, rgb: Image, box: BoundingBox) -> Gaze:
        crop = cv2.warpAffine(rgb, box_transform(box, _GAZE_SIZE), (_GAZE_SIZE, _GAZE_SIZE))
        blob = ((crop / np.float32(255) - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        yaw, pitch = self._gaze.run(
            None, {self._gaze.get_inputs()[0].name: blob.astype(np.float32)}
        )
        return gaze_from_bins(np.asarray(yaw, np.float32)[0], np.asarray(pitch, np.float32)[0])

    def _expressions(
        self, frame: Image, detections: Sequence[Detection]
    ) -> npt.NDArray[np.float32]:
        # FER2013 is grayscale: the model was fine-tuned on gray faces copied to three channels.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crops = np.stack(
            [
                cv2.warpAffine(
                    gray,
                    box_transform(detection.box, _EXPRESSION_SIZE),
                    (_EXPRESSION_SIZE, _EXPRESSION_SIZE),
                )
                for detection in detections
            ]
        )
        blob = np.repeat((crops / np.float32(127.5) - 1)[:, None], 3, axis=1).astype(np.float32)
        (logits,) = self._expression.run(None, {self._expression.get_inputs()[0].name: blob})
        return _softmax(np.asarray(logits, np.float32))


def mesh_transform(detection: Detection) -> npt.NDArray[np.float64]:
    """Affine map from the frame to the mesh model's crop: square, upright eyes, centered face."""
    right_eye, left_eye = np.array(detection.landmarks[0]), np.array(detection.landmarks[1])
    dx, dy = left_eye - right_eye
    box = detection.box
    center = (box.x + box.width / 2, box.y + box.height / 2)
    size = max(box.width, box.height) * _MESH_ROI_SCALE
    rotation = cv2.getRotationMatrix2D(center, math.degrees(math.atan2(dy, dx)), _MESH_SIZE / size)
    shift = np.array([_MESH_SIZE / 2 - center[0], _MESH_SIZE / 2 - center[1]])
    return np.asarray(rotation, np.float64) + np.column_stack([np.zeros((2, 2)), shift])


def box_transform(box: BoundingBox, size: int) -> npt.NDArray[np.float64]:
    """Affine map from the frame to a `size` by `size` crop of `box`."""
    scale_x, scale_y = size / box.width, size / box.height
    return np.array([[scale_x, 0, -box.x * scale_x], [0, scale_y, -box.y * scale_y]])


def gaze_from_bins(yaw: npt.NDArray[np.float32], pitch: npt.NDArray[np.float32]) -> Gaze:
    """Gaze angles from L2CS-Net's probabilities over 4-degree bins (softmax in the graph).

    L2CS-Net's yaw is positive towards the left of the image, as its demo draws it: it is flipped.
    Its pitch is already positive upwards.
    """
    bins = np.arange(len(yaw), dtype=np.float32)
    return Gaze(
        yaw=-float(yaw @ bins * _GAZE_BINS - 180),
        pitch=float(pitch @ bins * _GAZE_BINS - 180),
    )


def _softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
