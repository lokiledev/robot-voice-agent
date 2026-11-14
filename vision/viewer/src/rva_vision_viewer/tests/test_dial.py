import pytest

from rva_schemas.vision import CameraIntrinsics
from rva_vision_viewer.dial import describe, dial_point, view_edges


@pytest.mark.parametrize(
    ("bearing_deg", "point"),
    [
        (0.0, (0.0, -10.0)),  # Ahead: up the dial.
        (90.0, (10.0, 0.0)),  # The camera's right: the dial's.
        (-90.0, (-10.0, 0.0)),
        (180.0, (0.0, 10.0)),  # Behind: down.
    ],
)
def test_bearings_go_round_the_dial_as_seen_from_above(
    bearing_deg: float, point: tuple[float, float]
) -> None:
    assert dial_point(bearing_deg, 10.0) == pytest.approx(point, abs=1e-9)


def test_the_view_spans_the_image_either_side_of_its_centre() -> None:
    intrinsics = CameraIntrinsics(
        calibrated=True, width=1280, height=720, fx=640.0, fy=640.0, cx=640.0, cy=360.0
    )

    assert view_edges(intrinsics) == pytest.approx((-45.0, 45.0))


@pytest.mark.parametrize(
    ("bearing_deg", "text"),
    [(12.4, "Speech 12° right of the camera"), (-30.0, "Speech 30° left of the camera")],
)
def test_the_dial_says_where_speech_is(bearing_deg: float, text: str) -> None:
    assert describe(bearing_deg) == text
