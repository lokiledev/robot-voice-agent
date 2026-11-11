from datetime import UTC, datetime
from typing import Literal

from rva_interaction.config import EngagementConfig, InteractionConfig, SoundDirectionConfig
from rva_interaction.interaction import Interaction
from rva_interaction.tests.people import AT_ROBOT, AWAY, at, face, frame, seen
from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan, UserUtterance
from rva_schemas.frames import StaticTransforms
from rva_schemas.interaction import Engaged
from rva_schemas.record import Record
from rva_schemas.vision import Gaze

NOW = datetime(2026, 11, 11, tzinfo=UTC)
HERE = at(0.0, 1.5)


def config(barge_in: Literal["anyone", "engaged"]) -> InteractionConfig:
    return InteractionConfig(
        eye_contact_deg=15.0,
        history_s=30.0,
        min_speaking_share=0.3,
        min_segment_s=1.0,
        engaged_min_gaze=0.1,
        bystander_min_gaze=0.5,
        sound_direction=SoundDirectionConfig(tolerance_deg=15.0, max_age_s=0.2),
        barge_in=barge_in,
        engagement=EngagementConfig(
            near_m=2.0,
            far_m=4.5,
            moving_window_s=0.5,
            walking_m_s=1.0,
            rise_s=0.7,
            fall_s=3.0,
            enter=0.6,
            leave=0.2,
            max_engaged=3,
        ),
    )


class _NoRecorder:
    def record(self, entry: Record) -> None:
        pass


def interaction(barge_in: Literal["anyone", "engaged"] = "engaged") -> Interaction:
    return Interaction(
        config(barge_in), score_delay_s=0.0, transforms=StaticTransforms(()), recorder=_NoRecorder()
    )


def spoken(start: float, end: float) -> UserUtterance:
    return UserUtterance(
        text="Hello.",
        started_at=NOW,
        ended_at=NOW,
        speech=SpeechSpan(start=start, end=end),
        speaker=None,
    )


def talk(heard: Interaction, start: float, seconds: float, gaze: Gaze, *, beside: bool) -> None:
    """Track 1 speaks at 1.5 m, looking along `gaze`, ten frames a second from `start`.

    With `beside`, track 2 stands silent next to them, looking at them.
    """
    other = (seen(2, face(AWAY, at(0.8, 1.5)), speaking=False),) if beside else ()
    for i in range(round(seconds * 10)):
        heard.observe(frame(start + i / 10, seen(1, face(gaze, HERE), speaking=True), *other))


def test_a_bystander_who_looks_at_the_robot_while_speaking_talks_to_it_and_is_engaged() -> None:
    heard = interaction()
    # Looking elsewhere at first, so not engaged, then at the robot for the sentence.
    talk(heard, 0.0, 1.0, AWAY, beside=False)
    for i in range(3):
        heard.observe(frame(1.0 + i / 10, seen(1, face(AT_ROBOT, HERE), speaking=True)))
    assert heard.engaged() == frozenset()

    utterance = heard.hear(spoken(1.0, 1.2))

    assert utterance.speaker == SeenSpeaking(track_id=1, person=None, to_robot=True)
    assert heard.engaged() == frozenset({1})
    assert heard.observe(frame(1.3, seen(1, face(AT_ROBOT, HERE), speaking=False))).events == (
        Engaged(track_id=1, person=None),
    )


def test_a_bystander_who_speaks_looking_elsewhere_does_not_talk_to_the_robot() -> None:
    heard = interaction()
    talk(heard, 0.0, 1.0, AWAY, beside=False)

    assert heard.hear(spoken(0.0, 0.9)).speaker == SeenSpeaking(
        track_id=1, person=None, to_robot=False
    )
    assert heard.engaged() == frozenset()


def test_someone_engaged_with_nobody_else_in_view_talks_to_the_robot_wherever_they_look() -> None:
    heard = interaction()
    talk(heard, 0.0, 3.0, AT_ROBOT, beside=False)
    talk(heard, 3.0, 2.0, AWAY, beside=False)

    assert heard.hear(spoken(3.0, 4.9)).speaker == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )


def test_someone_engaged_beside_someone_else_talks_to_the_robot_even_glancing_at_it_only() -> None:
    heard = interaction()
    talk(heard, 0.0, 3.0, AT_ROBOT, beside=True)
    # A two-second sentence, looking at the robot for the first 0.4 s of it only.
    talk(heard, 3.0, 0.4, AT_ROBOT, beside=True)
    talk(heard, 3.4, 1.6, AWAY, beside=True)

    assert heard.hear(spoken(3.0, 4.9)).speaker == SeenSpeaking(
        track_id=1, person=None, to_robot=True
    )


def test_someone_engaged_who_faces_someone_else_throughout_does_not_talk_to_the_robot() -> None:
    heard = interaction()
    talk(heard, 0.0, 3.0, AT_ROBOT, beside=True)
    talk(heard, 3.0, 2.0, AWAY, beside=True)

    assert heard.hear(spoken(3.0, 4.9)).speaker == SeenSpeaking(
        track_id=1, person=None, to_robot=False
    )


def test_an_utterance_nobody_was_seen_speaking_is_from_nobody_in_view() -> None:
    heard = interaction()
    for i in range(10):
        heard.observe(frame(i / 10, seen(1, face(AT_ROBOT, HERE), speaking=False)))

    assert heard.hear(spoken(0.0, 0.9)).speaker == NobodySeenSpeaking()


def test_typed_lines_and_everything_without_an_active_speaker_pass_as_they_are() -> None:
    typed = UserUtterance(text="Hi.", started_at=NOW, ended_at=NOW, speech=None, speaker=None)
    deaf = Interaction(
        config("engaged"),
        score_delay_s=None,
        transforms=StaticTransforms(()),
        recorder=_NoRecorder(),
    )

    assert interaction().hear(typed) == typed
    assert deaf.hear(spoken(0.0, 1.0)) == spoken(0.0, 1.0)


def test_only_an_engaged_person_seen_speaking_may_interrupt_once_someone_is_engaged() -> None:
    heard = interaction()
    assert heard.may_interrupt()

    # Track 1 is engaged and silent; track 2, behind, talks to someone else.
    for i in range(20):
        heard.observe(
            frame(
                i / 10,
                seen(1, face(AT_ROBOT, HERE), speaking=False),
                seen(2, face(AWAY, at(1.0, 3.0)), speaking=True),
            )
        )
    assert heard.engaged() == frozenset({1})
    assert not heard.may_interrupt()

    heard.observe(frame(2.0, seen(1, face(AT_ROBOT, HERE), speaking=True)))
    assert heard.may_interrupt()


def test_anyone_may_interrupt_when_told_so() -> None:
    heard = interaction("anyone")
    for i in range(20):
        heard.observe(frame(i / 10, seen(1, face(AT_ROBOT, HERE), speaking=False)))

    assert heard.may_interrupt()
