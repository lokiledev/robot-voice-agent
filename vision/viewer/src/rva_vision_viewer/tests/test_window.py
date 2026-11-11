from uuid import UUID

import numpy as np

from rva_schemas.attention import FaceEntity, HandEntity
from rva_schemas.interaction import PersonEngagement
from rva_schemas.record import TrackedPerson
from rva_schemas.vision import (
    ActiveSpeaker,
    Body,
    BoundingBox,
    DepthMap,
    Hand,
    Keypoint,
    KnownPerson,
    MainUser,
)
from rva_vision_viewer.window import (
    candidate_name,
    depth_at,
    describe_body,
    describe_depth,
    describe_main_user,
    describe_person,
    fit,
    signals,
    title,
)


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
        match_score=0.8,
        confirmed=True,
        said="hello",
        face=None,
        body=None,
        speaker=None,
    )

    assert title(person) == '#3 Alice (3f2a1b4c): "hello"'
    assert title(person.model_copy(update={"person": None, "said": None})) == "#3 unknown"


def test_describe_body_counts_the_keypoints_found() -> None:
    point = Keypoint(x=1, y=2, score=0.9, position=None)
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


def test_the_main_user_is_described() -> None:
    user = MainUser(
        track_id=4,
        gaze_off_camera=6.4,
        eye_contact=True,
        mouth_motion=0.0412,
        lips_moving=True,
        active_speaker=ActiveSpeaker(score=-0.42, speaking=True),
    )
    unheard = user.model_copy(update={"lips_moving": False, "active_speaker": None})

    assert describe_main_user(user) == (
        "main user #4: eye contact, gaze 6 deg off; lips moving, motion 0.041; speaking (-0.4)"
    )
    assert describe_main_user(unheard) == (
        "main user #4: eye contact, gaze 6 deg off; lips still, motion 0.041; no microphone"
    )
    assert describe_main_user(None) == "nobody in view"


def _person(speaking: bool) -> TrackedPerson:
    return TrackedPerson(
        track_id=3,
        box=BoundingBox(x=0, y=0, width=10, height=10),
        person=None,
        match_score=None,
        confirmed=True,
        said=None,
        face=None,
        body=None,
        speaker=ActiveSpeaker(score=0.5, speaking=speaking),
    )


def test_a_persons_signals_are_off_for_nobody_and_while_their_engagement_is_unknown() -> None:
    engaged = PersonEngagement(
        track_id=3,
        closeness=0.8,
        staying=1.0,
        looking=True,
        attending=True,
        score=0.874,
        engaged=True,
    )

    assert signals(_person(speaking=True), engaged) == {
        "looking": True,
        "speaking": True,
        "engaged": True,
    }
    assert signals(_person(speaking=False), None) == {
        "looking": False,
        "speaking": False,
        "engaged": False,
    }
    assert signals(None, None) == {"looking": False, "speaking": False, "engaged": False}
    assert (
        describe_person(_person(speaking=True), engaged)
        == "engagement 0.87: closeness 0.80, staying 1.00"
    )
    assert describe_person(_person(speaking=True), None) == ""


def test_candidates_of_the_attention_are_named_by_what_they_are_and_whose() -> None:
    assert candidate_name(FaceEntity(track_id=3, person=None)) == "face #3"
    assert candidate_name(HandEntity(track_id=3, side="left")) == "left hand #3"
