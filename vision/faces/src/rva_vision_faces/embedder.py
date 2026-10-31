from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

from rva_vision_faces.detector import Detection, Image

type Embedding = npt.NDArray[np.float32]
"""A face descriptor of unit length, so that a dot product is a cosine similarity."""

ALIGNED_SIZE: Final = 112
# Where SFace expects the five landmarks in its 112x112 input (same as OpenCV's alignCrop).
_TEMPLATE: Final = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)


class SfaceEmbedder:
    """Face embeddings with SFace (opencv_zoo, 2021dec)."""

    def __init__(self, session: onnxruntime.InferenceSession) -> None:
        self._session: Final = session
        self._input: Final = session.get_inputs()[0].name

    def embed(self, frame: Image, detection: Detection) -> Embedding:
        face = align(frame, detection)
        # RGB, 0 to 255, channels first: what OpenCV's FaceRecognizerSF feeds the network.
        blob = face[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        (features,) = self._session.run(None, {self._input: blob})
        vector = np.asarray(features, np.float32)[0]
        return vector / np.linalg.norm(vector)


def align(frame: Image, detection: Detection) -> Image:
    """Crop the face and warp it so its landmarks fall on the template."""
    points = np.array(detection.landmarks, np.float32)
    transform, _ = cv2.estimateAffinePartial2D(points, _TEMPLATE, method=cv2.LMEDS)
    assert transform is not None, f"Degenerate landmarks: {detection.landmarks}"
    return np.asarray(cv2.warpAffine(frame, transform, (ALIGNED_SIZE, ALIGNED_SIZE)), np.uint8)
