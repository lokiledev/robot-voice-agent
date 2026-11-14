import math
from uuid import UUID

import pytest

from rva_attention.attention import VisualAttention
from rva_attention.config import FeatureWeights, HabituationConfig, VisualAttentionConfig
from rva_schemas.attention import AttentionState, FaceEntity, HandEntity
from rva_schemas.frames import CameraPose
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
        closeness=1.0,
        centrality=1.0,
        speaking=1.0,
        eye_contact=1.0,
        moving=2.0,
        known=0.5,
        novelty=1.0,
        raised=2.5,
    ),
    far_m=4.0,
    edge_deg=45.0,
    eye_contact_deg=20.0,
    moving_window_s=0.3,
    speed_m_s=0.5,
    novelty_decay_s=1.0,
    face=HabituationConfig(floor=0.3, decay_s=10.0, recovery_s=5.0),
    hand=HabituationConfig(floor=0.1, decay_s=1.0, recovery_s=3.0),
    # The weights add up to 10, which the score is over.
    margin=0.3 / 10,
    min_dwell_s=1.0,
    hold_s=0.3,
    min_score=0.2 / 10,
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


SHOULDERS_Y = 500.0
"""Where everyone's shoulders are in the image: a hand above, at a smaller y, is raised."""


def hand(point: Point3D, y: float) -> Hand:
    """A hand whose wrist and finger bases are at `point`, at `y` in the image."""
    keypoints: list[Keypoint | None] = [None] * 21
    for index in (0, 1, 5, 9, 13, 17):
        keypoints[index] = Keypoint(x=0.0, y=y, score=1.0, position=point)
    return Hand(keypoints=tuple(keypoints))


def body(right_hand: Hand) -> Body:
    shoulder = Keypoint(x=0.0, y=SHOULDERS_Y, score=1.0, position=None)
    keypoints = (None,) * 5 + (shoulder, shoulder) + (None,) * 6
    return Body(box=BOX, keypoints=keypoints, left_hand=None, right_hand=right_hand)


def person(
    track_id: int,
    point: Point3D | None,
    *,
    gaze: Gaze = STRAIGHT,
    known: KnownPerson | None = None,
    seen_speaking: bool | None = None,
    right_hand: Point3D | None = None,
    raised: bool = False,
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
        else body(hand(right_hand, SHOULDERS_Y - 200 if raised else SHOULDERS_Y + 200)),
        speaker=None if seen_speaking is None else ActiveSpeaker(score=1.0, speaking=seen_speaking),
    )


def observe(
    time: float, *people: TrackedPerson, camera: CameraPose | None = None
) -> VisualObservation:
    return VisualObservation(
        time=time,
        people=PeopleInView(
            frame="camera_optical",
            image_width=1920,
            image_height=1080,
            intrinsics=None,
            people=people,
            unmatched_bodies=(),
        ),
        camera_pose=camera,
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

    state = attention.observe(observe(0.0, person(7, IN_FRONT)), None)

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
        "novelty": 1.0,
        "raised": 0.0,
    }
    assert candidate.habituation == 1.0
    # Closeness 0.5, centrality 1, eye contact 1 and novelty 1, of 10.
    assert candidate.score == pytest.approx(3.5 / 10)


def test_nothing_is_attended_without_a_point_in_3d_or_before_a_track_is_confirmed() -> None:
    attention = VisualAttention(CONFIG)

    flat = attention.observe(observe(0.0, person(7, None)), None)
    unconfirmed = attention.observe(observe(0.1, person(8, IN_FRONT, confirmed=False)), None)

    assert flat.target is None and flat.candidates == ()
    assert unconfirmed.target is None and unconfirmed.candidates == ()


def test_a_closer_face_takes_over_once_the_first_was_attended_for_the_dwell() -> None:
    attention = VisualAttention(CONFIG)
    closer = Point3D(x=0.0, y=0.0, z=1.0)

    states = [
        attention.observe(
            observe(t / 10, person(1, IN_FRONT), *([person(2, closer)] if t >= 5 else [])), None
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


def test_a_raised_waving_hand_draws_attention_then_yields_to_the_face() -> None:
    attention = VisualAttention(CONFIG)
    states = []
    for t in range(60):
        # The hand hangs still at the hip, then waves above the shoulders from 3 to 5 s.
        up = 30 <= t < 50
        palm = Point3D(x=0.3 * (t % 2), y=-0.3, z=1.5) if up else Point3D(x=0.0, y=0.5, z=1.5)
        seen = person(1, IN_FRONT, right_hand=palm, raised=up)
        states.append(attention.observe(observe(t / 10, seen), None))

    assert changes(states) == [
        (0.0, FaceEntity(track_id=1, person=None)),
        (3.0, HandEntity(track_id=1, side="right")),
        (4.1, FaceEntity(track_id=1, person=None)),
    ]
    down, waving = states[29].candidates, states[31].candidates
    assert [c.entity.kind for c in down] == ["face", "hand"]
    assert down[1].features.raised == 0.0
    assert [c.entity.kind for c in waving] == ["hand", "face"]
    assert (waving[0].features.raised, waving[0].features.moving) == (1.0, 1.0)


def test_a_hand_moving_below_the_shoulders_does_not_draw_attention() -> None:
    attention = VisualAttention(CONFIG)
    states = []
    for t in range(30):
        # Talking with the hands: they move fast, low.
        palm = Point3D(x=0.3 * (t % 2), y=0.5, z=1.5)
        states.append(
            attention.observe(observe(t / 10, person(1, IN_FRONT, right_hand=palm)), None)
        )

    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None))]
    hand = states[-1].candidates[1]
    assert (hand.features.raised, hand.features.moving) == (0.0, 1.0)


def test_a_hand_coming_into_view_moving_low_does_not_draw_attention() -> None:
    # As someone turns, a hand the camera did not see appears, moving.
    attention = VisualAttention(CONFIG)
    states = []
    for t in range(40):
        palm = Point3D(x=0.3 * (t % 2), y=0.5, z=1.5) if t >= 20 else None
        states.append(
            attention.observe(observe(t / 10, person(1, IN_FRONT, right_hand=palm)), None)
        )

    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None))]
    hand = states[20].candidates[1]
    assert (hand.features.novelty, hand.features.moving) == (0.0, 0.0)


def test_everyone_s_hands_are_candidates() -> None:
    attention = VisualAttention(CONFIG)
    beside = Point3D(x=0.5, y=0.0, z=2.0)

    state = attention.observe(
        observe(
            0.0,
            person(1, IN_FRONT, right_hand=IN_FRONT),
            person(2, beside, right_hand=beside, raised=True),
        ),
        None,
    )

    assert {(c.entity.kind, c.entity.track_id) for c in state.candidates} == {
        ("face", 1),
        ("hand", 1),
        ("face", 2),
        ("hand", 2),
    }


def test_the_target_is_held_through_a_missed_frame_and_dropped_after_the_hold() -> None:
    attention = VisualAttention(CONFIG)
    seen = [True] * 6 + [False] + [True] * 3 + [False] * 5

    states = [
        attention.observe(observe(t / 10, *([person(1, IN_FRONT)] if there else [])), None)
        for t, there in enumerate(seen)
    ]

    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None)), (1.3, None)]
    missed = states[6]
    assert missed.candidates == ()
    assert missed.target is not None and missed.target.point == IN_FRONT


def test_a_face_that_leaves_is_dropped_at_once_and_comes_back_habituated() -> None:
    attention = VisualAttention(CONFIG)
    for t in range(20):
        attention.observe(observe(t / 10, person(1, IN_FRONT)), None)

    gone = attention.observe(observe(2.3), None)
    back = attention.observe(observe(2.5, person(1, IN_FRONT)), None)

    assert gone.target is None and gone.candidates == ()
    assert back.target is not None and back.target.entity == FaceEntity(track_id=1, person=None)
    [candidate] = back.candidates
    assert candidate.features.novelty == pytest.approx(0.082, abs=0.001)
    assert candidate.habituation == pytest.approx(0.86, abs=0.005)


def test_a_known_person_back_on_a_new_track_is_not_new_again() -> None:
    attention = VisualAttention(CONFIG)
    for t in range(20):
        attention.observe(observe(t / 10, person(1, IN_FRONT, known=ALICE)), None)
    attention.observe(observe(2.3), None)

    back = attention.observe(observe(2.5, person(2, IN_FRONT, known=ALICE)), None)

    assert back.target is not None
    assert back.target.entity == FaceEntity(track_id=2, person=ALICE)
    [candidate] = back.candidates
    assert candidate.features.novelty < 0.1
    assert candidate.habituation < 0.9
    assert candidate.features.known == 1.0


def test_a_forgotten_face_is_new_again() -> None:
    attention = VisualAttention(CONFIG)
    attention.observe(observe(0.0, person(1, IN_FRONT)), None)

    back = attention.observe(observe(2.5, person(1, IN_FRONT)), None)

    assert back.candidates[0].features.novelty == 1.0


def test_a_still_face_far_at_the_edge_looking_away_is_as_good_as_nobody() -> None:
    attention = VisualAttention(CONFIG)
    far_aside = Point3D(x=4.0, y=0.0, z=4.0)
    away = Gaze(yaw=40.0, pitch=0.0)

    states = [
        attention.observe(observe(t / 10, person(1, far_aside, gaze=away)), None) for t in range(31)
    ]

    assert states[0].target is not None, "new, it is worth a glance"
    assert states[-1].target is None
    # Its novelty alone, fading as it is attended: under 0.2 of the 10 after 1.6 s.
    assert changes(states) == [(0.0, FaceEntity(track_id=1, person=None)), (1.6, None)]
    [candidate] = states[-1].candidates
    assert candidate.features.closeness == 0.0
    assert candidate.features.centrality == 0.0
    assert candidate.features.eye_contact == 0.0
    # A novelty of 0.05 after 3 s, of 10, a little habituated still.
    assert candidate.features.novelty == pytest.approx(0.05, abs=0.001)
    assert candidate.score == pytest.approx(0.0046, abs=0.0002)


def test_speaking_and_moving_count() -> None:
    attention = VisualAttention(CONFIG)
    states = [
        attention.observe(observe(t / 10, person(1, Point3D(x=0.1 * t, y=0.0, z=2.0))), 1)
        for t in range(4)
    ]

    features = states[-1].candidates[0].features
    assert features.speaking == 1.0
    assert features.moving == 1.0, "1 m/s is over the 0.5 m/s scale"


def test_only_the_face_speaking_counts_as_speaking_whoever_else_is_seen_to() -> None:
    attention = VisualAttention(CONFIG)
    beside = Point3D(x=0.5, y=0.0, z=2.0)

    state = attention.observe(
        observe(
            0.0,
            person(1, IN_FRONT, seen_speaking=True),
            person(2, beside, seen_speaking=True),
        ),
        2,
    )

    speaking = {c.entity.track_id: c.features.speaking for c in state.candidates}
    assert speaking == {1: 0.0, 2: 1.0}


def test_the_face_attended_keeps_the_head_while_it_speaks_however_habituated() -> None:
    # Two people side by side; whoever has been attended for 8 s speaks, the other listens. The
    # active speaker sees both speak; the direction of the speech tells the speaker.
    attention = VisualAttention(CONFIG)
    left, right = Point3D(x=-0.3, y=0.0, z=2.0), Point3D(x=0.3, y=0.0, z=2.0)
    states = []
    for t in range(120):
        speaker = None
        if t >= 80:
            attended = states[79].target
            assert attended is not None
            speaker = attended.entity.track_id
        both = (person(1, left, seen_speaking=t >= 80), person(2, right, seen_speaking=t >= 80))
        states.append(attention.observe(observe(t / 10, *both), speaker))

    before = states[79].target
    assert before is not None
    assert [s.target.entity for s in states[80:] if s.target is not None] == [before.entity] * 40


def test_a_still_face_does_not_move_as_the_camera_turns_past_it() -> None:
    # A face 2 m ahead of the robot, the head turning left at 1 radian a second.
    ahead = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))

    def turned(yaw: float) -> CameraPose:
        c, s = math.cos(yaw), math.sin(yaw)
        turn = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

        def row(i: int) -> tuple[float, float, float]:
            a, b, d = (sum(turn[i][k] * ahead[k][j] for k in range(3)) for j in range(3))
            return (a, b, d)

        return CameraPose(position=(0.0, 0.0, 0.0), rotation=(row(0), row(1), row(2)))

    def seen_from(camera: CameraPose) -> Point3D:
        # The face at (2, 0, 0) in the fixed frame, in the camera's: by the rotation's transpose,
        # whose rows are the rotation's first entries of each row.
        r = camera.rotation
        return Point3D(x=2.0 * r[0][0], y=2.0 * r[0][1], z=2.0 * r[0][2])

    cameras = [turned(0.1 * t) for t in range(4)]
    followed, unaware = VisualAttention(CONFIG), VisualAttention(CONFIG)
    with_pose = [
        followed.observe(observe(t / 10, person(1, seen_from(c)), camera=c), None)
        for t, c in enumerate(cameras)
    ][-1]
    without = [
        unaware.observe(observe(t / 10, person(1, seen_from(c))), None)
        for t, c in enumerate(cameras)
    ][-1]

    assert with_pose.candidates[0].features.moving == pytest.approx(0.0, abs=1e-9)
    assert without.candidates[0].features.moving == 1.0, "2 m/s across the image"
    # The target is still where the camera sees it.
    assert with_pose.candidates[0].point == seen_from(cameras[-1])
