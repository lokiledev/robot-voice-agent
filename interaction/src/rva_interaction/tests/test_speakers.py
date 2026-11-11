import math
from uuid import UUID

from rva_interaction.config import SoundDirectionConfig
from rva_interaction.speakers import SpeakerAttribution
from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan
from rva_schemas.frames import FrameTransform, StaticTransforms
from rva_schemas.observation import VisualObservation
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.sound import HeardDirection
from rva_schemas.vision import ActiveSpeaker, BoundingBox, CameraIntrinsics, KnownPerson

ALICE = KnownPerson(user_id=UUID("00000000-0000-0000-0000-000000000001"), name="Alice")
BOX = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
UNDER_CAMERA = FrameTransform(
    parent="camera_optical",
    child="respeaker",
    xyz=(0.0, 0.05, 0.0),
    rpy=(math.pi / 2, -math.pi / 2, 0.0),
)
"""The array lies flat just under the camera: its x axis straight ahead, its y axis to the left."""
WIDE = CameraIntrinsics(
    calibrated=True, width=1000, height=500, fx=500.0, fy=500.0, cx=500.0, cy=250.0
)
"""90 degrees across: a face centred at pixel u is atan((u - 500) / 500) right of the axis."""
LEFT = BoundingBox(x=200.0, y=100.0, width=100.0, height=100.0)
"""Centred at 250: 26.6 degrees left."""
RIGHT = BoundingBox(x=700.0, y=100.0, width=100.0, height=100.0)
"""Centred at 750: 26.6 degrees right."""


def attributing(score_delay_s: float) -> SpeakerAttribution:
    return SpeakerAttribution(
        history_s=30.0,
        min_share=0.3,
        min_segment_s=1.0,
        score_delay_s=score_delay_s,
        sound=SoundDirectionConfig(tolerance_deg=15.0, max_age_s=0.2),
        transforms=StaticTransforms((UNDER_CAMERA,)),
    )


def tracked(
    track_id: int, *, speaking: bool, person: KnownPerson | None = None, confirmed: bool = True
) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=BOX,
        person=person,
        match_score=None,
        confirmed=confirmed,
        said=None,
        face=None,
        body=None,
        speaker=ActiveSpeaker(score=1.0 if speaking else -3.0, speaking=speaking),
    )


def at(person: TrackedPerson, box: BoundingBox) -> TrackedPerson:
    return person.model_copy(update={"box": box})


def surely(person: TrackedPerson, score: float) -> TrackedPerson:
    """`person` seen speaking, with the active speaker this sure of it."""
    return person.model_copy(update={"speaker": ActiveSpeaker(score=score, speaking=True)})


def frame(time: float, *people: TrackedPerson) -> VisualObservation:
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
        main_user=None,
    )


def wide(time: float, *people: TrackedPerson) -> VisualObservation:
    """A frame of the `WIDE` camera."""
    return VisualObservation(
        time=time,
        people=PeopleInView(
            frame="camera_optical",
            image_width=1000,
            image_height=500,
            intrinsics=WIDE,
            people=people,
            unmatched_bodies=(),
        ),
        main_user=None,
    )


def heard(time: float, bearing_deg: float | None) -> HeardDirection:
    """Speech from `bearing_deg` right of the camera, the array's minus that; none with `None`."""
    return HeardDirection(
        time=time,
        frame="respeaker",
        speech_azimuth=None if bearing_deg is None else math.radians(-bearing_deg),
    )


def test_an_utterance_is_said_by_the_face_seen_speaking_for_most_of_it() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Ten frames a second for a second: track 2 speaks throughout, track 1 only nods along twice.
    for i in range(10):
        attribution.observe(
            frame(i / 10, tracked(1, speaking=i < 2), tracked(2, speaking=True, person=ALICE))
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=2, person=ALICE, to_robot=True
    )


def test_only_what_was_seen_during_the_utterance_counts() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1 speaks for the first second, track 2 for the next.
    for i in range(20):
        attribution.observe(
            frame(i / 10, tracked(1, speaking=i < 10), tracked(2, speaking=i >= 10))
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )
    assert attribution.attribute(SpeechSpan(start=1.0, end=1.9)) == SeenSpeaking(
        track_id=2, person=None, to_robot=True
    )


def test_a_frame_tells_who_spoke_a_little_before_it() -> None:
    attribution = attributing(score_delay_s=0.5)
    # Seen speaking in the frames from 1.0 to 1.9 s, so spoken from 0.5 to 1.4 s.
    for i in range(30):
        attribution.observe(frame(i / 10, tracked(1, speaking=10 <= i < 20)))

    assert attribution.attribute(SpeechSpan(start=0.5, end=1.4)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )
    assert attribution.attribute(SpeechSpan(start=1.5, end=2.4)) == NobodySeenSpeaking()


def test_nobody_said_it_when_no_face_spoke_for_enough_of_it() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1's lips move in 2 frames of 10, under the 0.3 it takes.
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=i < 2)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_nobody_said_it_when_nothing_was_seen_meanwhile() -> None:
    attribution = attributing(score_delay_s=0.0)
    attribution.observe(frame(0.0, tracked(1, speaking=True)))

    assert attribution.attribute(SpeechSpan(start=5.0, end=6.0)) == NobodySeenSpeaking()


def test_a_face_not_yet_confirmed_says_nothing() -> None:
    attribution = attributing(score_delay_s=0.0)
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=True, confirmed=False)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_the_speaker_is_who_the_face_was_last_recognized_as() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Recognized halfway through the sentence.
    for i in range(10):
        attribution.observe(
            frame(i / 10, tracked(1, speaking=True, person=ALICE if i >= 5 else None))
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=ALICE, to_robot=True
    )


def test_what_was_seen_is_forgotten_after_the_history() -> None:
    attribution = attributing(score_delay_s=0.0)
    attribution.observe(frame(0.0, tracked(1, speaking=True)))
    attribution.observe(frame(31.0, tracked(1, speaking=False)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.5)) == NobodySeenSpeaking()


def test_among_faces_seen_speaking_only_the_surest_counts() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1 talks; track 2, listening, is seen speaking too all along, less surely.
    listener = tracked(2, speaking=True)
    listener = listener.model_copy(update={"speaker": ActiveSpeaker(score=-0.5, speaking=True)})
    for i in range(10):
        attribution.observe(frame(i / 10, listener, tracked(1, speaking=True)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )
    assert attribution.segments(SpeechSpan(start=0.0, end=0.9)) == (SpeechSpan(start=0.0, end=0.9),)


def test_a_turn_is_cut_where_the_speaker_changes() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1 for 2 s, track 2 for 1.5 s, track 1 again for 1.5 s.
    for i in range(50):
        first = i < 20 or i >= 35
        attribution.observe(
            frame(i / 10, tracked(1, speaking=first), tracked(2, speaking=not first))
        )

    assert attribution.segments(SpeechSpan(start=0.0, end=4.9)) == (
        SpeechSpan(start=0.0, end=2.0),
        SpeechSpan(start=2.0, end=3.5),
        SpeechSpan(start=3.5, end=4.9),
    )


def test_a_word_thrown_in_and_a_pause_do_not_cut_a_turn() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1 talks for 4 s, but for 0.5 s of silence from 1 s and track 2's "yeah" from 2.5 s.
    for i in range(40):
        attribution.observe(
            frame(
                i / 10,
                tracked(1, speaking=not (10 <= i < 15 or 25 <= i < 28)),
                tracked(2, speaking=25 <= i < 28),
            )
        )

    assert attribution.segments(SpeechSpan(start=0.0, end=3.9)) == (SpeechSpan(start=0.0, end=3.9),)


def test_a_short_start_by_someone_else_joins_the_piece_after_it() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 2 for 0.3 s, then track 1 for 2 s, then track 2 for 2 s.
    for i in range(43):
        second = i < 3 or i >= 23
        attribution.observe(
            frame(i / 10, tracked(1, speaking=not second), tracked(2, speaking=second))
        )

    assert attribution.segments(SpeechSpan(start=0.0, end=4.2)) == (
        SpeechSpan(start=0.0, end=2.3),
        SpeechSpan(start=2.3, end=4.2),
    )


def test_a_turn_nobody_was_seen_speaking_is_one_piece() -> None:
    attribution = attributing(score_delay_s=0.0)
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=False)))

    assert attribution.segments(SpeechSpan(start=0.0, end=0.9)) == (SpeechSpan(start=0.0, end=0.9),)


def test_the_direction_heard_tells_the_talker_from_a_listener_seen_speaking() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1, on the left, talks; track 2, on the right, nods along and looks surer to the lips.
    talker, listener = (
        at(tracked(1, speaking=True), LEFT),
        surely(at(tracked(2, speaking=True), RIGHT), 2.0),
    )
    for i in range(10):
        attribution.hear(heard(i / 10, -26.6))
        attribution.observe(wide(i / 10, talker, listener))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )


def test_speech_from_out_of_view_is_nobodys_in_view() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Someone 80 degrees to the right talks; track 1, in view, moves its lips along.
    for i in range(10):
        attribution.hear(heard(i / 10, 80.0))
        attribution.observe(wide(i / 10, at(tracked(1, speaking=True), LEFT)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_a_face_heard_but_not_seen_speaking_said_it() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Track 1 talks turned away, so its lips are not seen; track 2 is silent.
    for i in range(10):
        attribution.hear(heard(i / 10, -20.0))
        attribution.observe(
            wide(
                i / 10, at(tracked(1, speaking=False), LEFT), at(tracked(2, speaking=False), RIGHT)
            )
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )


def test_speech_heard_from_between_two_silent_faces_is_nobodys() -> None:
    attribution = attributing(score_delay_s=0.0)
    # 10 degrees apart, both within 15 degrees of the speech at 5.
    straight = BoundingBox(x=450.0, y=100.0, width=100.0, height=100.0)
    ten_right = BoundingBox(x=538.2, y=100.0, width=100.0, height=100.0)
    for i in range(10):
        attribution.hear(heard(i / 10, 5.0))
        attribution.observe(
            wide(
                i / 10,
                at(tracked(1, speaking=False), straight),
                at(tracked(2, speaking=False), ten_right),
            )
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_the_lips_win_when_speech_comes_from_where_no_face_is() -> None:
    attribution = attributing(score_delay_s=0.0)
    # Heard straight ahead, 26.6 degrees from the only face, which is seen speaking.
    for i in range(10):
        attribution.hear(heard(i / 10, 0.0))
        attribution.observe(wide(i / 10, at(tracked(1, speaking=True), LEFT)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )


def test_the_lips_decide_when_no_speech_is_heard() -> None:
    attribution = attributing(score_delay_s=0.0)
    talker, listener = (
        at(tracked(1, speaking=True), LEFT),
        surely(at(tracked(2, speaking=True), RIGHT), 2.0),
    )
    for i in range(10):
        attribution.hear(heard(i / 10, None))
        attribution.observe(wide(i / 10, talker, listener))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=2, person=None, to_robot=True
    )


def test_a_direction_heard_too_long_ago_is_not_used() -> None:
    attribution = attributing(score_delay_s=0.0)
    talker, listener = (
        at(tracked(1, speaking=True), LEFT),
        surely(at(tracked(2, speaking=True), RIGHT), 2.0),
    )
    # Heard from the left once, then nothing for the 0.2 s a direction holds.
    attribution.hear(heard(0.0, -26.6))
    for i in range(10):
        attribution.observe(wide(1 + i / 10, talker, listener))

    assert attribution.attribute(SpeechSpan(start=1.0, end=1.9)) == SeenSpeaking(
        track_id=2, person=None, to_robot=True
    )


def test_without_a_microphone_array_the_lips_decide() -> None:
    attribution = attributing(score_delay_s=0.0)
    talker, listener = (
        at(tracked(1, speaking=True), LEFT),
        surely(at(tracked(2, speaking=True), RIGHT), 2.0),
    )
    # No direction is ever heard.
    for i in range(10):
        attribution.observe(wide(i / 10, talker, listener))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=2, person=None, to_robot=True
    )
