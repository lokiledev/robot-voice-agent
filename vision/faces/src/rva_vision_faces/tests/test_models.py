import numpy as np
import pytest

from rva_schemas.vision import BoundingBox
from rva_vision_faces.config import DetectorConfig, ModelFile
from rva_vision_faces.detector import Detection, YunetDetector
from rva_vision_faces.embedder import SfaceEmbedder
from rva_vision_faces.models import load_session

YUNET = ModelFile(repo="opencv/face_detection_yunet", file="face_detection_yunet_2023mar.onnx")
SFACE = ModelFile(repo="opencv/face_recognition_sface", file="face_recognition_sface_2021dec.onnx")


@pytest.mark.download
def test_yunet_runs_on_any_frame_size_and_finds_no_face_in_a_blank_frame() -> None:
    detector = YunetDetector(
        load_session(YUNET, "cpu", dynamic_image_size=True),
        DetectorConfig(model=YUNET, score_threshold=0.8, nms_threshold=0.3),
    )

    assert detector.detect(np.full((1080, 1920, 3), 128, np.uint8)) == []
    assert detector.detect(np.full((480, 650, 3), 128, np.uint8)) == []


@pytest.mark.download
def test_sface_gives_a_unit_embedding() -> None:
    embedder = SfaceEmbedder(load_session(SFACE, "cpu", dynamic_image_size=False))
    frame = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    detection = Detection(
        box=BoundingBox(x=100, y=100, width=100, height=120),
        landmarks=((130, 140), (170, 140), (150, 165), (135, 190), (165, 190)),
        score=0.9,
    )

    embedding = embedder.embed(frame, detection)

    assert embedding.shape == (128,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
