from uuid import UUID

import numpy as np

from rva_schemas.record import TrackedPerson
from rva_schemas.vision import Body, BoundingBox, DepthMap, Hand, Keypoint, KnownPerson
from rva_vision_viewer.window import depth_at, describe_body, describe_depth, fit, title


def test_fit_keeps_the_aspect_ratio_within_the_space() -> None:
    assert fit((960, 540), (1920, 1080)) == (1920, 1080)
    assert fit((960, 540), (480, 1000)) == (480, 270)
    assert fit((960, 540), (2000, 270)) == (480, 270)
    assert fit((960, 540), (-10, 300)) == (1, 1)


def test_title_names_the_person_and_what_they_said() -> None:
    alice = KnownPerson(user_id=UUID("3f2a1b4c-0000-4000-8000-000000000001"), name="Alice")
    person = TrackedPerson(
        track_id=3,
        box=BoundingBox(x=0, y=0, width=10, height=10),
        person=alice,
        confirmed=True,
        said="hello",
        face=None,
        body=None,
    )

    assert title(person) == '#3 Alice (3f2a1b4c): "hello"'
    assert title(person.model_copy(update={"person": None, "said": None})) == "#3 unknown"


def test_describe_body_counts_the_keypoints_found() -> None:
    point = Keypoint(x=1, y=2, score=0.9)
    body = Body(
        box=BoundingBox(x=0, y=0, width=10, height=10),
        keypoints=(point, None, point),
        left_hand=Hand(keypoints=(point, None)),
        right_hand=None,
    )

    assert describe_body(body) == "skeleton 2/3, hands left 1/2 right -"


def test_depth_at_takes_the_median_around_a_point() -> None:
    depth = np.full((10, 20), 2.0, np.float32)
    depth[0, 0] = 9.0
    depth[9, 19] = 0.5

    assert depth_at(depth, (0.5, 0.5)) == 2.0
    assert depth_at(depth, (0.0, 0.0)) == 2.0
    assert depth_at(depth, (1.5, -1.0)) == 2.0


def test_describe_depth_says_whether_it_is_calibrated() -> None:
    depth = np.full((10, 20), 1.234, np.float32)

    assert (
        describe_depth(DepthMap(metres=depth, calibrated=True), (0.5, 0.5))
        == "depth 1.23 m (calibrated)"
    )
    assert (
        describe_depth(DepthMap(metres=depth, calibrated=False), (0.5, 0.5))
        == "depth 1.23 m (UNCALIBRATED)"
    )
