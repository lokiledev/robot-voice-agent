from uuid import UUID

from rva_interaction.speakers import SpeakerAttribution
from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan
from rva_schemas.observation import VisualObservation
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import ActiveSpeaker, BoundingBox, KnownPerson

ALICE = KnownPerson(user_id=UUID("00000000-0000-0000-0000-000000000001"), name="Alice")
BOX = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)


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


def frame(time: float, *people: TrackedPerson) -> VisualObservation:
    return VisualObservation(
        time=time,
        people=PeopleInView(
            image_width=1920, image_height=1080, intrinsics=None, people=people, unmatched_bodies=()
        ),
        main_user=None,
    )


def test_an_utterance_is_said_by_the_face_seen_speaking_for_most_of_it() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    # Ten frames a second for a second: track 2 speaks throughout, track 1 only nods along twice.
    for i in range(10):
        attribution.observe(
            frame(i / 10, tracked(1, speaking=i < 2), tracked(2, speaking=True, person=ALICE))
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=2, person=ALICE, to_robot=True
    )


def test_only_what_was_seen_during_the_utterance_counts() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
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
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.5
    )
    # Seen speaking in the frames from 1.0 to 1.9 s, so spoken from 0.5 to 1.4 s.
    for i in range(30):
        attribution.observe(frame(i / 10, tracked(1, speaking=10 <= i < 20)))

    assert attribution.attribute(SpeechSpan(start=0.5, end=1.4)) == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )
    assert attribution.attribute(SpeechSpan(start=1.5, end=2.4)) == NobodySeenSpeaking()


def test_nobody_said_it_when_no_face_spoke_for_enough_of_it() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    # Track 1's lips move in 2 frames of 10, under the 0.3 it takes.
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=i < 2)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_nobody_said_it_when_nothing_was_seen_meanwhile() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    attribution.observe(frame(0.0, tracked(1, speaking=True)))

    assert attribution.attribute(SpeechSpan(start=5.0, end=6.0)) == NobodySeenSpeaking()


def test_a_face_not_yet_confirmed_says_nothing() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=True, confirmed=False)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == NobodySeenSpeaking()


def test_the_speaker_is_who_the_face_was_last_recognized_as() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    # Recognized halfway through the sentence.
    for i in range(10):
        attribution.observe(
            frame(i / 10, tracked(1, speaking=True, person=ALICE if i >= 5 else None))
        )

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.9)) == SeenSpeaking(
        track_id=1, person=ALICE, to_robot=True
    )


def test_what_was_seen_is_forgotten_after_the_history() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    attribution.observe(frame(0.0, tracked(1, speaking=True)))
    attribution.observe(frame(31.0, tracked(1, speaking=False)))

    assert attribution.attribute(SpeechSpan(start=0.0, end=0.5)) == NobodySeenSpeaking()


def test_among_faces_seen_speaking_only_the_surest_counts() -> None:
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
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
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
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
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
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
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
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
    attribution = SpeakerAttribution(
        history_s=30.0, min_share=0.3, min_segment_s=1.0, score_delay_s=0.0
    )
    for i in range(10):
        attribution.observe(frame(i / 10, tracked(1, speaking=False)))

    assert attribution.segments(SpeechSpan(start=0.0, end=0.9)) == (SpeechSpan(start=0.0, end=0.9),)
