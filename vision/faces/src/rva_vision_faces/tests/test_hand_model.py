import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.transform import Rotation

from rva_schemas.vision import CameraIntrinsics
from rva_vision_faces.hand_model import PARAMS, Params, Side, fit, hand_points

INTRINSICS = CameraIntrinsics(
    calibrated=False, width=1920, height=1080, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
)
WRIST = np.array([0.05, 0.1, 0.6])


def _hand(turn: tuple[float, float, float], bends: tuple[float, float, float, float]) -> Params:
    """A hand turned by `turn`, in radians, from palm to the camera and fingers up.

    Each finger is bent by `bends`: base, spread, middle and end.
    """
    params = np.zeros(PARAMS)
    palm_to_camera = Rotation.from_euler("x", math.pi).as_matrix()
    params[:3] = Rotation.from_matrix(
        palm_to_camera @ Rotation.from_euler("xyz", turn).as_matrix()
    ).as_rotvec()
    params[4:] = np.tile(bends, 5)
    return params


def _seen(points: npt.NDArray[np.float64]) -> dict[int, tuple[float, float]]:
    return {
        index: (1000 * x / z + 960, 1000 * y / z + 540)
        for index, (x, y, z) in enumerate(points.tolist())
    }


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_hand_is_found_again_from_its_keypoints(side: Side) -> None:
    points = WRIST + hand_points(_hand((0.3, -0.4, 0.2), (0.5, 0.1, 0.6, 0.3)), side)

    fitted, _ = fit(_seen(points), WRIST, side, INTRINSICS, 1.0, None)

    assert np.abs(fitted - points).max() == pytest.approx(0.0, abs=2e-3)


def test_a_finger_s_missing_joints_are_placed_along_it() -> None:
    points = WRIST + hand_points(_hand((0.2, 0.0, 0.0), (0.4, 0.0, 0.4, 0.2)), "right")
    seen = _seen(points)
    # The index finger's two middle joints were not found.
    del seen[6], seen[7]

    fitted, _ = fit(seen, WRIST, "right", INTRINSICS, 1.0, None)

    assert np.abs(fitted[5:9] - points[5:9]).max() == pytest.approx(0.0, abs=5e-3)


def test_a_misplaced_keypoint_does_not_drag_the_hand() -> None:
    points = WRIST + hand_points(_hand((0.2, 0.0, 0.0), (0.4, 0.0, 0.4, 0.2)), "right")
    seen = _seen(points)
    # The ring finger's tip, 80 pixels off.
    x, y = seen[16]
    seen[16] = (x + 80, y)

    fitted, _ = fit(seen, WRIST, "right", INTRINSICS, 1.0, None)

    # It still pulls a little: within a centimetre, where it would drag them by several.
    others = [index for index in range(21) if index not in (14, 15, 16)]
    assert np.abs(fitted[others] - points[others]).max() == pytest.approx(0.0, abs=1e-2)


def test_fingers_bend_only_within_their_range() -> None:
    # Keypoints of fingers bent a radian backwards, which no finger can.
    bent_back = WRIST + hand_points(_hand((0.0, 0.0, 0.0), (-1.0, 0.0, -1.0, 0.0)), "right")

    _, params = fit(_seen(bent_back), WRIST, "right", INTRINSICS, 1.0, None)

    fingers = params[4:].reshape(5, 4)
    assert fingers[1:, 0].min() >= -0.35
    assert fingers[1:, 2].min() >= 0.0


def test_the_last_frame_s_fit_is_followed() -> None:
    first = _hand((0.2, 0.0, 0.0), (0.4, 0.0, 0.4, 0.2))
    _, params = fit(
        _seen(WRIST + hand_points(first, "right")), WRIST, "right", INTRINSICS, 1.0, None
    )
    moved = _hand((0.25, 0.05, 0.0), (0.5, 0.0, 0.5, 0.3))
    points = WRIST + hand_points(moved, "right")

    fitted, _ = fit(_seen(points), WRIST, "right", INTRINSICS, 1.0, params)

    assert np.abs(fitted - points).max() == pytest.approx(0.0, abs=5e-3)


def test_a_palm_squeezed_into_a_few_pixels_is_still_fitted() -> None:
    # A far or misdetected hand, its palm's keypoints within two pixels of each other.
    seen = {index: (500.0 + index % 2, 300.0) for index in (0, 1, 5, 9, 13, 17)}

    fitted, _ = fit(seen, WRIST, "left", INTRINSICS, 1.0, None)

    assert fitted[0] == pytest.approx(WRIST)
