import math

import pytest

from rva_schemas.frames import FrameTransform, MissingTransformError, StaticTransforms

UNDER_CAMERA = FrameTransform(
    parent="camera_optical",
    child="respeaker",
    xyz=(0.0, 0.05, 0.0),
    rpy=(math.pi / 2, -math.pi / 2, 0.0),
)
"""An array lying flat under a camera: x straight ahead, y to the left, z up."""


def test_a_child_axis_is_turned_into_the_parents_frame() -> None:
    # The camera's optical frame has x right, y down and z ahead.
    assert UNDER_CAMERA.rotate((1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert UNDER_CAMERA.rotate((0.0, 1.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)
    assert UNDER_CAMERA.rotate((0.0, 0.0, 1.0)) == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


def test_unrotating_undoes_rotating() -> None:
    turned = FrameTransform(parent="a", child="b", xyz=(0.0, 0.0, 0.0), rpy=(0.3, -0.7, 1.9))

    assert turned.unrotate(turned.rotate((0.2, -0.5, 0.8))) == pytest.approx((0.2, -0.5, 0.8))


def test_a_transform_is_found_either_way_round() -> None:
    transforms = StaticTransforms((UNDER_CAMERA,))

    assert transforms.rotate(
        (1.0, 0.0, 0.0), source="respeaker", target="camera_optical"
    ) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert transforms.rotate(
        (0.0, 0.0, 1.0), source="camera_optical", target="respeaker"
    ) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert transforms.connects("camera_optical", "respeaker")
    assert transforms.connects("respeaker", "camera_optical")


def test_links_not_given_a_transform_between_them_are_not_connected() -> None:
    transforms = StaticTransforms((UNDER_CAMERA,))

    assert not transforms.connects("camera_optical", "base_link")
    with pytest.raises(MissingTransformError):
        transforms.rotate((1.0, 0.0, 0.0), source="base_link", target="camera_optical")
