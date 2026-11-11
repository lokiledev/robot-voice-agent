import math

import pytest

from rva_interaction.bearings import apart, face_bearing, speech_bearing, view_bearings
from rva_schemas.frames import FrameTransform, StaticTransforms
from rva_schemas.sound import HeardDirection
from rva_schemas.vision import BoundingBox, CameraIntrinsics

WIDE = CameraIntrinsics(
    calibrated=True, width=1000, height=500, fx=500.0, fy=500.0, cx=500.0, cy=250.0
)
TRANSFORMS = StaticTransforms(
    (
        # Flat under the camera: x straight ahead, y to the left.
        FrameTransform(
            parent="camera_optical",
            child="respeaker",
            xyz=(0.0, 0.05, 0.0),
            rpy=(math.pi / 2, -math.pi / 2, 0.0),
        ),
    )
)


def test_a_face_is_where_its_box_is_centred() -> None:
    box = BoundingBox(x=900.0, y=0.0, width=200.0, height=200.0)

    assert face_bearing(box, WIDE) == pytest.approx(45.0)


def test_speech_is_turned_into_the_cameras_frame() -> None:
    left = HeardDirection(time=0.0, frame="respeaker", speech_azimuth=math.pi / 2)
    ahead_right = HeardDirection(time=0.0, frame="respeaker", speech_azimuth=-math.pi / 6)

    assert speech_bearing(left, "camera_optical", TRANSFORMS) == pytest.approx(-90.0)
    assert speech_bearing(ahead_right, "camera_optical", TRANSFORMS) == pytest.approx(30.0)


def test_no_speech_has_no_bearing() -> None:
    silence = HeardDirection(time=0.0, frame="respeaker", speech_azimuth=None)

    assert speech_bearing(silence, "camera_optical", TRANSFORMS) is None


def test_the_view_spans_the_image_edges() -> None:
    assert view_bearings(WIDE) == pytest.approx((-45.0, 45.0))


def test_bearings_are_apart_the_short_way_round() -> None:
    assert apart(170.0, -170.0) == pytest.approx(20.0)
    assert apart(-10.0, 25.0) == pytest.approx(35.0)
