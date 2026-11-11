from uuid import UUID

from rva_schemas.interaction import PersonEngagement
from rva_schemas.record import TrackedPerson
from rva_schemas.vision import BoundingBox, KnownPerson
from rva_vision_viewer.overlay import color_of, describe_engagement, label

ALICE = KnownPerson(user_id=UUID("12345678-0000-0000-0000-000000000001"), name="Alice")


def person(
    track_id: int, who: KnownPerson | None, score: float | None, confirmed: bool
) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
        person=who,
        match_score=score,
        confirmed=confirmed,
        said=None,
        face=None,
        body=None,
        speaker=None,
    )


def test_label_shows_the_track_the_name_the_user_id_and_the_match() -> None:
    assert label(person(1, ALICE, 0.734, True)) == "#1 Alice (12345678) 0.73"
    assert label(person(2, None, None, True)) == "#2 unknown"


def test_color_is_gray_until_announced_then_green_if_recognized_else_orange() -> None:
    assert color_of(person(1, ALICE, 0.7, False)) == (160, 160, 160)
    assert color_of(person(1, ALICE, 0.7, True)) == (80, 200, 80)
    assert color_of(person(1, None, None, True)) == (0, 165, 255)


def test_engagement_is_described_with_its_score() -> None:
    engaged = PersonEngagement(
        track_id=1, closeness=1.0, staying=1.0, attending=True, score=0.874, engaged=True
    )

    assert describe_engagement(engaged) == "engaged 0.87"
    assert (
        describe_engagement(engaged.model_copy(update={"engaged": False, "score": 0.3}))
        == "not engaged 0.30"
    )
