from uuid import UUID

import pytest

from rva_attention.attention import VisualAttention
from rva_attention.config import (
    FeatureWeights,
    HabituationConfig,
    NoveltyConfig,
    VisualAttentionConfig,
)
from rva_schemas.attention import AttentionState, FaceEntity, HandEntity
from rva_schemas.observation import VisualObservation
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import (
    ActiveSpeaker,
    Body,
    BoundingBox,
    Face3D,
    FaceFeatures,
    Gaze,
    GazeRay,
    Hand,
    Keypoint,
    KnownPerson,
    Point3D,
)

CONFIG = VisualAttentionConfig(
    weights=FeatureWeights(
        closeness=1.0, centrality=1.0, speaking=1.0, eye_contact=1.0, moving=2.0, known=0.5
    ),
    far_m=4.0,
    edge_deg=45.0,
    eye_contact_deg=20.0,
    moving_window_s=0.3,
    speed_m_s=0.5,
    novelty=NoveltyConfig(amplitude=1.0, decay_s=1.0),
    face=HabituationConfig(floor=0.3, decay_s=10.0, recovery_s=5.0),
    hand=HabituationConfig(floor=0.1, decay_s=1.0, recovery_s=3.0),
    margin=0.3,
    min_dwell_s=1.0,
    hold_s=0.3,
    min_score=0.2,
    lifetime_s=2.0,
)
ALICE = KnownPerson(user_id=UUID("00000000-0000-0000-0000-000000000001"), name="Alice")
IN_FRONT = Point3D(x=0.0, y=0.0, z=2.0)
BOX = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
STRAIGHT = Gaze(yaw=0.0, pitch=0.0)


def face(point: Point3D | None, gaze: Gaze) -> FaceFeatures:
    """A face whose every 3D landmark, the irises included, is at `point`."""
    return FaceFeatures(
        landmarks=((0.0, 0.0),) * 478,
        left_eye_open=1.0,
        right_eye_open=1.0,
        mouth_open=0.0,
        gaze=gaze,
        in_3d=None
        if point is None
        else Face3D(
            landmarks=(point,) * 478,
            gaze=GazeRay(origin=point, direction=Point3D(x=0.0, y=0.0, z=-1.0)),
        ),
        expression="neutral",
        expression_score=1.0,
    )


def hand(point: Point3D) -> Hand:
    """A hand whose wrist and finger bases are at `point`."""
    keypoints: list[Keypoint | None] = [None] * 21
    for index in (0, 1, 5, 9, 13, 17):
        keypoints[index] = Keypoint(x=0.0, y=0.0, score=1.0, position=point)
    return Hand(keypoints=tuple(keypoints))


def person(
    track_id: int,
    point: Point3D | None,
    *,
    gaze: Gaze = STRAIGHT,
    known: KnownPerson | None = None,
    speaking: bool | None = None,
    right_hand: Point3D | None = None,
    confirmed: bool = True,
) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=BOX,
        person=known,
        match_score=None if known is None else 0.9,
        confirmed=confirmed,
        said=None,
        face=face(point, gaze),
        body=None
        if right_hand is None
        else Body(box=BOX, keypoints=(None,) * 13, left_hand=None, right_hand=hand(right_hand)),
        speaker=None if speaking is None else ActiveSpeaker(score=1.0, speaking=speaking),
    )


def observe(time: float, *people: TrackedPerson, main_user: int | None = None) -> VisualObservation:
    return VisualObservation(
        time=time,
        people=PeopleInView(
            image_width=1920, image_height=1080, intrinsics=None, people=people, unmatched_bodies=()
        ),
        main_user=main_user,
    )


def changes(states: list[AttentionState]) -> list[tuple[float, FaceEntity | HandEntity | None]]:
    """When the target changed, and to what."""
    changed = []
    previous: object = object()
    for state in states:
        entity = None if state.target is None else state.target.entity
        if entity != previous:
            changed.append((state.time, entity))
            previous = entity
    return changed


def test_the_one_face_in_view_is_attended_between_its_irises() -> None:
    attention = VisualAttention(CONFIG)

    state = attention.observe(observe(0.0, person(7, IN_FRONT)))

    assert state.target is not None
    assert state.target.entity == FaceEntity(track_id=7, person=None)
    assert state.target.point == IN_FRONT
    [candidate] = state.candidates
    assert candidate.features.model_dump() == {
        "closeness": 0.5,
        "centrality": 1.0,
        "speaking": 0.0,
        "eye_contact": 1.0,
        "moving": 0.0,
        "known": 0.0,
    }
    assert candidate.novelty == 1.0
    assert candidate.habituation == 1.0
    assert candidate.score == pytest.approx(3.5)


def test_nothing_is_attended_without_a_point_in_3d_or_before_a_track_is_confirmed() -> None:
    attention = VisualAttention(CONFIG)

    flat = attention.observe(observe(0.0, person(7, None)))
    unconfirmed = attention.observe(observe(0.1, person(8, IN_FRONT, confirmed=False)))

    assert flat.target is None and flat.candidates == ()
    assert unconfirmed.target is None and unconfirmed.candidates == ()


def test_a_closer_face_takes_over_once_the_first_was_attended_for_the_dwell() -> None:
    attention = VisualAttention(CONFIG)
    closer = Point3D(x=0.0, y=0.0, z=1.0)

    states = [
        attention.observe(
            observe(t / 10, person(1, IN_FRONT), *([person(2, closer)] if t >= 5 else []))
        )
        for t in range(16)
    ]

    assert changes(states) == [
        (0.0, FaceEntity(track_id=1, person=None)),
        (1.0, FaceEntity(track_id=2, person=None)),
    ]
    at_half = states[5]
    assert [c.entity.track_id for c in at_half.candidates] == [2, 1]
    assert at_half.candidates[0].score > at_half.candidates[1].score + CONFIG.margin


def test_a_waving_hand_draws_attention_then_yields_to_the_face() -> None:
    attention = VisualAttention(CONFIG)
    states = []
    for t in range(60):
        time = t / 10
        waving = Point3D(x=0.3 * (t % 2), y=0.0, z=1.5) if t >= 30 else None
        states.append(
            attention.observe(observe(time, person(1, IN_FRONT, right_hand=waving), main_user=1))
        )

    assert changes(states) == [
        (0.0, FaceEntity(track_id=1, person=None)),
        (3.1, HandEntity(track_id=1, side="right")),
        (4.1, FaceEntity(track_id=1, person=None)),
    ]
    just_waving = states[32]
    assert [c.entity.kind for c in just_waving.candidates] == ["hand", "face"]
    assert just_waving.candidates[0].features.moving == 1.0


def test_a_hand_of_someone_else_than_the_main_user_is_no_candidate() -> None:
    attention = VisualAttention(CONFIG)

    state = attention.observe(observe(0.0, person(1, IN_FRONT, right_hand=IN_FRONT), main_user=2))

    assert [c.entity.kind for c in state.candidates] == ["face"]


def test_the_target_is_held_through_a_missed_frame_and_dropped_after_the_hold() -> None:
    attention = VisualAttention(CONFIG)
    seen = [True] * 6 + [False] + [True] * 3 + [False] * 5

    states = [
        attention.observe(observe(t / 10, *([person(1, IN_FRONT)] if there else [])))
        for t, there in enumerate(seen)
    ]

    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None)), (1.3, None)]
    missed = states[6]
    assert missed.candidates == ()
    assert missed.target is not None and missed.target.point == IN_FRONT


def test_a_face_that_leaves_is_dropped_at_once_and_comes_back_habituated() -> None:
    attention = VisualAttention(CONFIG)
    for t in range(20):
        attention.observe(observe(t / 10, person(1, IN_FRONT)))

    gone = attention.observe(observe(2.3))
    back = attention.observe(observe(2.5, person(1, IN_FRONT)))

    assert gone.target is None and gone.candidates == ()
    assert back.target is not None and back.target.entity == FaceEntity(track_id=1, person=None)
    [candidate] = back.candidates
    assert candidate.novelty == pytest.approx(0.082, abs=0.001)
    assert candidate.habituation == pytest.approx(0.86, abs=0.005)


def test_a_known_person_back_on_a_new_track_is_not_new_again() -> None:
    attention = VisualAttention(CONFIG)
    for t in range(20):
        attention.observe(observe(t / 10, person(1, IN_FRONT, known=ALICE)))
    attention.observe(observe(2.3))

    back = attention.observe(observe(2.5, person(2, IN_FRONT, known=ALICE)))

    assert back.target is not None
    assert back.target.entity == FaceEntity(track_id=2, person=ALICE)
    [candidate] = back.candidates
    assert candidate.novelty < 0.1
    assert candidate.habituation < 0.9
    assert candidate.features.known == 1.0


def test_a_forgotten_face_is_new_again() -> None:
    attention = VisualAttention(CONFIG)
    attention.observe(observe(0.0, person(1, IN_FRONT)))

    back = attention.observe(observe(2.5, person(1, IN_FRONT)))

    assert back.candidates[0].novelty == 1.0


def test_a_still_face_far_at_the_edge_looking_away_is_as_good_as_nobody() -> None:
    attention = VisualAttention(CONFIG)
    far_aside = Point3D(x=4.0, y=0.0, z=4.0)
    away = Gaze(yaw=40.0, pitch=0.0)

    states = [
        attention.observe(observe(t / 10, person(1, far_aside, gaze=away))) for t in range(31)
    ]

    assert states[0].target is not None, "new, it is worth a glance"
    assert states[-1].target is None
    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None)), (1.7, None)]
    [candidate] = states[-1].candidates
    assert candidate.features.closeness == 0.0
    assert candidate.features.centrality == 0.0
    assert candidate.features.eye_contact == 0.0
    assert candidate.score == pytest.approx(0.05, abs=0.001)


def test_speaking_and_moving_count() -> None:
    attention = VisualAttention(CONFIG)
    states = [
        attention.observe(
            observe(t / 10, person(1, Point3D(x=0.1 * t, y=0.0, z=2.0), speaking=True))
        )
        for t in range(4)
    ]

    features = states[-1].candidates[0].features
    assert features.speaking == 1.0
    assert features.moving == 1.0, "1 m/s is over the 0.5 m/s scale"
