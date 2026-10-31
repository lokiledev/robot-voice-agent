from uuid import UUID

from pydantic import TypeAdapter

from rva_schemas.vision import (
    BoundingBox,
    FaceAppeared,
    FaceEvent,
    FaceLeft,
    KnownPerson,
    TrackedFace,
)

EVENTS = TypeAdapter(list[FaceEvent])
ALICE = KnownPerson(user_id=UUID("3f2a1b4c-0000-4000-8000-000000000001"), name="Alice")


def test_face_events_deserialize() -> None:
    serialized = (
        '[{"kind": "appeared", "face": {"track_id": 7, "box": {"x": 10.0, "y": 20.0, "width": 150.0,'
        ' "height": 180.0}, "person": {"user_id": "3f2a1b4c-0000-4000-8000-000000000001",'
        ' "name": "Alice"}, "nearest": true}},'
        ' {"kind": "left", "track_id": 8, "person": null}]'
    )

    assert EVENTS.validate_json(serialized) == [
        FaceAppeared(
            face=TrackedFace(
                track_id=7,
                box=BoundingBox(x=10.0, y=20.0, width=150.0, height=180.0),
                person=ALICE,
                nearest=True,
            )
        ),
        FaceLeft(track_id=8, person=None),
    ]
