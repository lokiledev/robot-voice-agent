"""A kinematic hand, fitted to a hand's keypoints in the image.

The hand is a rigid palm, the wrist and the base of each finger, with four joints per finger:
the base bends and spreads, the two others only bend, each within a finger's range. Fitted to
RTMW's 21 keypoints in the image, with the wrist held at a given point, it places the hand in
3D from its 2D keypoints alone: the depth model, which sees the room through fingers, is asked
for nothing but how far the wrist is.

The hand's frame, for a right hand: x towards the thumb, y along the fingers, z out of the palm;
a left hand is its mirror. A hand's points are in metres in that frame, the wrist at the origin.
"""

import math
from typing import Final, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rva_schemas.vision import CameraIntrinsics

type Side = Literal["left", "right"]
type Params = npt.NDArray[np.float64]
"""The rotation of the hand as a rotation vector, the log of its size against the average hand,
then four angles per finger, in radians, from the thumb: base bend, base spread, middle bend,
end bend."""

_PALM_SCALE: Final = 1.45
"""RTMW's wrist and finger bases are farther apart than an anatomical palm's, against the
fingers: by 1.33 to 1.56 on three hands of two people, measured at the depth."""
_THUMB_BASE_SCALE: Final = 1.2
_PALM: Final = {
    1: (0.025 * _THUMB_BASE_SCALE, 0.030 * _THUMB_BASE_SCALE, 0.005),
    5: (0.025 * _PALM_SCALE, 0.090 * _PALM_SCALE, 0.0),
    9: (0.003 * _PALM_SCALE, 0.092 * _PALM_SCALE, 0.0),
    13: (-0.017 * _PALM_SCALE, 0.086 * _PALM_SCALE, 0.0),
    17: (-0.035 * _PALM_SCALE, 0.076 * _PALM_SCALE, 0.0),
}
"""Base of each finger in a right hand's frame, of an average adult hand as RTMW places them."""
_FINGERS: Final = (
    # base, rest direction in the palm (radians towards the thumb), bone lengths in metres
    (1, 0.80, (0.045, 0.032, 0.027)),
    (5, 0.10, (0.042, 0.025, 0.022)),
    (9, 0.0, (0.046, 0.028, 0.023)),
    (13, -0.10, (0.042, 0.027, 0.022)),
    (17, -0.20, (0.033, 0.019, 0.020)),
)
"""Each finger from the thumb: its base keypoint, the direction it points to at rest, and its
three bones, the next three keypoints at their ends."""
_LIMITS: Final = (
    ((-0.5, 1.0), (-0.6, 0.6), (-0.2, 1.2), (-0.3, 1.4)),
    *[((-0.35, 1.6), (-0.35, 0.35), (0.0, 1.9), (0.0, 1.4))] * 4,
)
"""Range of each finger's four angles, in radians: fingers bend towards the palm, barely back."""
_SIZE_LIMITS: Final = (0.5, 2.0)
_SMOOTHING_PX: Final = 10.0
"""Cost in pixels of moving a joint by a radian from the last frame's angle: the image decides,
but bends it cannot see stay put."""
_SIZE_PX: Final = 1000.0
"""Cost in pixels of a hand a factor e larger or smaller than it is held to, 100 for a tenth. The
fit trades size for bends where the keypoints are hard to explain, as a fist's: replayed on a
recording, a tenth of this let the size swing from 0.7 to 1.2 and jump by 85% between frames;
this holds it within 0.97 to 1.06, the keypoints 9 to 10 pixels off the hand at the median
instead of 7."""
_REST_PX: Final = 2.0
"""Cost in pixels of a joint a radian away from rest, without a last frame."""
_OUTLIER_PX: Final = 8.0
"""Keypoints off the fitted hand by more than this weigh less, for a misplaced one."""
_MIN_PALM_PX: Final = 10.0
"""Spread of a palm's keypoints in the image below which its turn is not guessed from them."""
_MAX_STEPS: Final = 30
"""Evaluations per fit and start: from the last frame's fit, a few steps reach it."""
_TOLERANCE: Final = 1e-3
"""Relative change of the cost at which the fit stops: keypoints are a few pixels off anyway."""
_STEP: Final = 1e-6
"""Of the finite differences the fit's derivatives are taken from."""
_ANGLES: Final = 4 * len(_FINGERS)
PARAMS: Final = 4 + _ANGLES

_LOWER: Final = np.array(
    [-np.inf] * 3 + [math.log(_SIZE_LIMITS[0])] + [low for finger in _LIMITS for low, _ in finger]
)
_UPPER: Final = np.array(
    [np.inf] * 3 + [math.log(_SIZE_LIMITS[1])] + [high for finger in _LIMITS for _, high in finger]
)


_BASES: Final = np.array([finger[0] for finger in _FINGERS])
_RESTS: Final = np.array([finger[1] for finger in _FINGERS])
_LENGTHS: Final = np.array([finger[2] for finger in _FINGERS])
_TEMPLATE: Final = np.zeros((21, 3))
_TEMPLATE[list(_PALM)] = list(_PALM.values())
_JOINTS: Final = np.array([[base + 1, base + 2, base + 3] for base in _BASES])
"""The three keypoints past each finger's base, from the thumb."""
_CURLED: Final = np.tile([0.8, 0.0, 0.8, 0.5], len(_FINGERS))


def hand_points(params: Params, side: Side) -> npt.NDArray[np.float64]:
    """The 21 points of the hand of `params`, rotated and sized, the wrist at the origin."""
    return _hands(params[None], side)[0]


def _hands(params: npt.NDArray[np.float64], side: Side) -> npt.NDArray[np.float64]:
    """`hand_points` of N hands at once, N x 21 x 3: the fit's derivatives take one call."""
    angles = params[:, 4:].reshape(len(params), len(_FINGERS), 4)
    toward = _RESTS + angles[:, :, 1]
    along = np.stack([np.sin(toward), np.cos(toward), np.zeros_like(toward)], axis=-1)
    bent = np.cumsum(angles[:, :, [0, 2, 3]], axis=2)
    # Each bone: its length along the finger's direction, bent towards the palm's normal.
    steps = _LENGTHS[None, :, :, None] * (
        np.cos(bent)[..., None] * along[:, :, None, :]
        + np.sin(bent)[..., None] * np.array([0.0, 0.0, 1.0])
    )
    points = np.repeat(_TEMPLATE[None], len(params), axis=0)
    points[:, _JOINTS] = _TEMPLATE[_BASES][None, :, None, :] + np.cumsum(steps, axis=2)
    if side == "left":
        points[:, :, 0] *= -1
    rotated = np.einsum("nij,npj->npi", _rotations(params[:, :3]), points)
    return np.exp(params[:, 3])[:, None, None] * rotated


def _rotations(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rodrigues' formula for N rotation vectors, N x 3 x 3."""
    angles = np.linalg.norm(vectors, axis=1)
    axes = vectors / np.maximum(angles, 1e-12)[:, None]
    x, y, z = axes.T
    zero = np.zeros_like(x)
    cross = np.stack(
        [np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)],
        axis=1,
    )
    sin, cos = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + sin * cross + (1 - cos) * cross @ cross


def fit(
    pixels: dict[int, tuple[float, float]],
    wrist: npt.NDArray[np.float64],
    side: Side,
    intrinsics: CameraIntrinsics,
    size: float,
    previous: Params | None,
) -> tuple[npt.NDArray[np.float64], Params]:
    """The hand whose points, with its wrist at `wrist`, best fall on its keypoints.

    The palm and the fingers are fitted together: the fingers tell which way a palm is turned,
    which its flat points alone leave open. Without a last frame, it is fitted from a flat hand
    and from a curled one, keeping the better.

    Args:
        pixels: The keypoints found, by index, in frame pixels; the wrist's among them.
        wrist: Where the wrist is in the camera's frame, on its keypoint's ray.
        side: Which hand it is, the person's own.
        intrinsics: The camera at the frame's resolution.
        size: The size against the average hand the hand is held near, e.g. its size so far.
        previous: The fit of the same hand in the last frame, to start from and stay near.

    Returns:
        The 21 points in the camera's frame, and the fit, for the next frame.
    """
    indices = [index for index in pixels if index != 0]
    observed = np.array([pixels[index] for index in indices], np.float64).reshape(-1, 2)
    anchor = previous if previous is not None else np.zeros(PARAMS)
    weight = _SMOOTHING_PX if previous is not None else _REST_PX

    def batch(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The residuals of N fits, N x residuals."""
        points = wrist + _hands(params, side)[:, indices]
        z = np.maximum(points[:, :, 2], 1e-3)
        error = np.stack(
            [
                intrinsics.fx * points[:, :, 0] / z + intrinsics.cx - observed[:, 0],
                intrinsics.fy * points[:, :, 1] / z + intrinsics.cy - observed[:, 1],
            ],
            axis=-1,
        )
        prior = np.column_stack(
            [_SIZE_PX * (params[:, 3] - math.log(size)), weight * (params[:, 4:] - anchor[4:])]
        )
        return np.concatenate([_robust(error.reshape(len(params), -1)), prior], axis=1)

    def residuals(params: Params) -> npt.NDArray[np.float64]:
        return batch(params[None])[0]

    def jacobian(params: Params) -> npt.NDArray[np.float64]:
        # Forward differences, backward at an upper bound, all in one call.
        step = np.where(params + _STEP <= _UPPER, _STEP, -_STEP)
        moved = params + np.diag(step)
        return ((batch(moved) - residuals(params)) / step[:, None]).T

    if previous is not None:
        starts = [previous]
    else:
        flat = _first_guess(pixels, side, intrinsics, size)
        starts = [flat, np.concatenate([flat[:4], _CURLED])]
    solutions = [
        least_squares(
            residuals,
            np.clip(start, _LOWER + 1e-9, _UPPER - 1e-9),
            bounds=(_LOWER, _UPPER),
            jac=jacobian,
            max_nfev=_MAX_STEPS,
            ftol=_TOLERANCE,
        )
        for start in starts
    ]
    params = np.asarray(min(solutions, key=lambda solution: solution.cost).x, np.float64)
    return wrist + hand_points(params, side), params


def _robust(error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Errors in pixels scaled so that their squares sum to a pseudo-Huber cost.

    Quadratic up to `_OUTLIER_PX`, linear past it, for a misplaced keypoint to pull less. The
    priors are left quadratic: scipy's own robust loss would soften them too.
    """
    ratio = error / _OUTLIER_PX
    return np.sign(error) * _OUTLIER_PX * np.sqrt(2 * (np.sqrt(1 + ratio**2) - 1))


def _first_guess(
    pixels: dict[int, tuple[float, float]], side: Side, intrinsics: CameraIntrinsics, size: float
) -> Params:
    """A flat hand of `size`, turned as its palm's keypoints are."""
    params = np.zeros(PARAMS)
    params[3] = math.log(size)
    # Palm towards the camera, fingers up, without enough of the palm to turn it.
    params[:3] = Rotation.from_euler("x", math.pi).as_rotvec()
    palm = [index for index in (0, *_PALM) if index in pixels]
    seen = np.array([pixels[index] for index in palm], np.float64).reshape(-1, 2)
    # PnP needs four points, spread out: OpenCV's asserts on a palm squeezed into a few pixels,
    # which says nothing of its turn anyway.
    if len(palm) < 4 or np.ptp(seen, axis=0).max() < _MIN_PALM_PX:
        return params
    camera = np.array(
        [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]]
    )
    found, rotation, _ = cv2.solvePnP(
        hand_points(np.zeros(PARAMS), side)[palm],
        seen,
        camera,
        None,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if found:
        params[:3] = rotation.ravel()
    return params
