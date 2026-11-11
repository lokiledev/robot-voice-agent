import pytest

from rva_interaction.config import EngagementConfig
from rva_interaction.engagement import EngagementTracker
from rva_interaction.tests.people import AT_ROBOT, AWAY, at, face, frame, seen
from rva_schemas.interaction import Disengaged, Engaged, EngagementEvent
from rva_schemas.observation import VisualObservation
from rva_schemas.vision import Gaze, Point3D

CONFIG = EngagementConfig(
    near_m=2.0,
    far_m=4.5,
    moving_window_s=0.5,
    walking_m_s=1.0,
    rise_s=0.7,
    fall_s=3.0,
    enter=0.6,
    leave=0.2,
    max_engaged=1,
)


def watch(tracker: EngagementTracker, frames: list[VisualObservation]) -> list[EngagementEvent]:
    return [event for observation in frames for event in tracker.observe(observation).events]


def standing(
    seconds: float, gaze: Gaze, point: Point3D | None, start: float
) -> list[VisualObservation]:
    """Track 1 standing at `point`, ten frames a second from `start`."""
    return [
        frame(start + i / 10, seen(1, face(gaze, point), speaking=False))
        for i in range(round(seconds * 10))
    ]


def test_someone_close_and_looking_is_engaged_within_a_second() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)

    states = [tracker.observe(o) for o in standing(1.0, AT_ROBOT, at(0.0, 1.5), start=0.0)]

    # 0.7 s to two thirds of the way: 0.6 is reached after 0.7 * ln(2.5) = 0.64 s.
    assert [state.time for state in states if state.events] == [0.7]
    assert states[7].events == (Engaged(track_id=1, person=None),)
    assert tracker.engaged == frozenset({1})


def test_someone_close_who_does_not_look_is_not_engaged() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)

    assert watch(tracker, standing(5.0, AWAY, at(0.0, 1.5), start=0.0)) == []


def test_someone_looking_from_too_far_is_not_engaged() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)

    # At 3.5 m the closeness is 0.4, under the 0.6 it takes.
    states = [tracker.observe(o) for o in standing(10.0, AT_ROBOT, at(0.0, 3.5), start=0.0)]

    assert [event for state in states for event in state.events] == []
    assert states[-1].people[0].closeness == pytest.approx(0.4)
    assert states[-1].people[0].score == pytest.approx(0.4, abs=0.01)


def test_someone_walking_past_while_looking_is_not_engaged() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    # Across the view at 1.2 m/s, 1.5 m away, glancing at the robot all along.
    walking = [
        frame(i / 10, seen(1, face(AT_ROBOT, at(-1.5 + 0.12 * i, 1.5)), speaking=False))
        for i in range(25)
    ]

    assert watch(tracker, walking) == []


def test_a_gap_between_frames_is_not_a_second_of_eye_contact() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    looking = seen(1, face(AT_ROBOT, at(0.0, 1.5)), speaking=False)
    # Seen in the first frame; the next comes 1.2 s later, as the models load.
    tracker.observe(frame(0.0, looking))

    state = tracker.observe(frame(1.2, looking))

    # It counts for 0.2 s: 1 - exp(-0.2 / 0.7).
    assert state.events == ()
    assert state.people[0].score == pytest.approx(0.249, abs=0.001)


def test_a_face_engages_only_once_confirmed() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    unconfirmed = seen(1, face(AT_ROBOT, at(0.0, 1.5)), speaking=False).model_copy(
        update={"confirmed": False}
    )

    assert watch(tracker, [frame(i / 10, unconfirmed) for i in range(30)]) == []
    assert tracker.observe(
        frame(3.0, unconfirmed.model_copy(update={"confirmed": True}))
    ).events == (Engaged(track_id=1, person=None),)


def test_looking_away_for_a_while_disengages_but_a_glance_away_does_not() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    here = at(0.0, 1.5)
    watch(tracker, standing(5.0, AT_ROBOT, here, start=0.0))

    assert watch(tracker, standing(2.0, AWAY, here, start=5.0)) == []
    assert watch(tracker, standing(3.0, AT_ROBOT, here, start=7.0)) == []
    # From 1 down to 0.2 takes 3 * ln(5) = 4.8 s.
    assert watch(tracker, standing(6.0, AWAY, here, start=10.0)) == [
        Disengaged(track_id=1, person=None, left=False)
    ]
    assert tracker.engaged == frozenset()


def test_someone_engaged_who_speaks_while_looking_away_stays_engaged() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    here = at(0.0, 1.5)
    watch(tracker, standing(5.0, AT_ROBOT, here, start=0.0))
    thinking_aloud = [
        frame(5.0 + i / 10, seen(1, face(AWAY, here), speaking=True)) for i in range(100)
    ]

    assert watch(tracker, thinking_aloud) == []


def test_someone_engaged_who_goes_out_of_view_has_left() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    watch(tracker, standing(5.0, AT_ROBOT, at(0.0, 1.5), start=0.0))

    assert watch(tracker, [frame(5.0)]) == [Disengaged(track_id=1, person=None, left=True)]


def test_no_more_people_are_engaged_than_allowed() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    both = [
        frame(
            i / 10,
            seen(1, face(AT_ROBOT, at(-0.5, 1.5)), speaking=False),
            seen(2, face(AT_ROBOT, at(0.5, 1.5)), speaking=False),
        )
        for i in range(50)
    ]

    assert watch(tracker, both) == [Engaged(track_id=1, person=None)]


def test_without_depth_looking_is_enough() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)

    assert watch(tracker, standing(1.0, AT_ROBOT, None, start=0.0)) == [
        Engaged(track_id=1, person=None)
    ]


def test_someone_who_spoke_to_the_robot_is_engaged_at_once() -> None:
    tracker = EngagementTracker(CONFIG, eye_contact_deg=15.0)
    far = at(0.0, 4.0)
    watch(tracker, standing(1.0, AT_ROBOT, far, start=0.0))

    tracker.engage(1)

    assert tracker.engaged == frozenset({1})
    assert watch(tracker, standing(0.1, AT_ROBOT, far, start=1.0)) == [
        Engaged(track_id=1, person=None)
    ]
