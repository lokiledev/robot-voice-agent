import numpy as np
import pytest

from rva_schemas.vision import BoundingBox
from rva_vision_faces.body import BodyEstimator
from rva_vision_faces.config import (
    BodyConfig,
    CameraIntrinsics,
    DetectorConfig,
    EyeContactConfig,
    FaceFeaturesConfig,
    ModelFile,
)
from rva_vision_faces.depth import DepthEstimator
from rva_vision_faces.detector import Detection, YunetDetector
from rva_vision_faces.embedder import SfaceEmbedder
from rva_vision_faces.face_features import FaceAnalyzer
from rva_vision_faces.models import load_session

YUNET = ModelFile(repo="opencv/face_detection_yunet", file="face_detection_yunet_2023mar.onnx")
SFACE = ModelFile(repo="opencv/face_recognition_sface", file="face_recognition_sface_2021dec.onnx")

DEPTH = ModelFile(repo="Heliosoph/da3metric-large-onnx", file="model_fp16.onnx")


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


FACE_FEATURES = FaceFeaturesConfig(
    landmarks=ModelFile(
        repo="fernandotonon/QtMeshEditor-facemesh-onnx", file="face_landmarks.onnx"
    ),
    blendshapes=ModelFile(
        repo="fernandotonon/QtMeshEditor-faceblendshapes-onnx", file="face_blendshapes.onnx"
    ),
    gaze=ModelFile(repo="mlboydaisuke/L2CS-Gaze360-OpenVINO", file="gaze_fp16.onnx"),
    expression=ModelFile(
        repo="onnx-community/face-emotion-detection-ONNX", file="onnx/model_fp16.onnx"
    ),
    eye_contact=EyeContactConfig(window_s=0.4, enter_deg=12.0, leave_deg=18.0),
)
BODY = BodyConfig(
    detector=ModelFile(repo="yzd-v/DWPose", file="yolox_l.onnx"),
    score_threshold=0.5,
    nms_threshold=0.45,
    pose=ModelFile(repo="bukuroo/RTMW-ONNX", file="rtmw-l-256.onnx"),
    keypoint_threshold=0.3,
)


@pytest.mark.download
def test_face_analyzer_gives_every_feature_of_each_face() -> None:
    analyzer = FaceAnalyzer(
        *(
            load_session(model, "cpu", dynamic_image_size=False)
            for model in (
                FACE_FEATURES.landmarks,
                FACE_FEATURES.blendshapes,
                FACE_FEATURES.gaze,
                FACE_FEATURES.expression,
            )
        )
    )
    frame = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    detection = Detection(
        box=BoundingBox(x=100, y=100, width=100, height=120),
        landmarks=((130, 140), (170, 140), (150, 165), (135, 190), (165, 190)),
        score=0.9,
    )

    (features,) = analyzer.analyze(frame, [detection])

    assert len(features.landmarks) == 478
    assert 0 <= features.left_eye_open <= 1
    assert 0 <= features.mouth_open <= 1
    assert -180 <= features.gaze.yaw <= 180
    assert 0 < features.expression_score <= 1


@pytest.mark.download
def test_body_estimator_finds_nobody_in_a_blank_frame() -> None:
    estimator = BodyEstimator(
        load_session(BODY.detector, "cpu", dynamic_image_size=False),
        load_session(BODY.pose, "cpu", dynamic_image_size=False),
        BODY,
    )

    assert estimator.estimate(np.full((720, 1280, 3), 128, np.uint8)) == []


@pytest.mark.download
def test_depth_keeps_the_frame_aspect_ratio_and_says_whether_it_is_calibrated() -> None:
    session = load_session(DEPTH, "cpu", dynamic_image_size=False)
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    intrinsics = CameraIntrinsics(
        calibrated=True, width=1920, height=1080, fx=1360, fy=1360, cx=960, cy=540
    )

    raw = DepthEstimator(session, None).estimate(frame)
    calibrated = DepthEstimator(session, intrinsics).estimate(frame)

    assert raw.metres.shape == (284, 504)
    assert not raw.calibrated
    assert calibrated.calibrated
    np.testing.assert_allclose(calibrated.metres, raw.metres * (1360 * 504 / 1920 / 300), rtol=1e-3)
