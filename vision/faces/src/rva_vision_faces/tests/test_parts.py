import math
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from rva_schemas.record import FaceBox, FaceBoxes
from rva_schemas.vision import (
    BoundingBox,
    FaceAppeared,
    FaceLeft,
    KnownPerson,
    NameTakenError,
    TrackedFace,
    UnknownPersonError,
)
from rva_vision_faces.config import MatchConfig, TrackerConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.detector import Detection, decode_yunet
from rva_vision_faces.embedder import Embedding, align
from rva_vision_faces.gallery import Gallery
from rva_vision_faces.tracker import Track, Tracker
from rva_vision_faces.vision import face_boxes
from rva_vision_faces.window import label

MATCHING = MatchConfig(threshold=0.363, margin=0.05)
TRACKING = TrackerConfig(
    min_face_height=0.12,
    iou_threshold=0.3,
    reid_threshold=0.5,
    confirm_after_s=0.5,
    leave_after_s=3.0,
    embed_every_s=1.0,
    embeddings_per_track=5,
)


def _person(name: str) -> KnownPerson:
    return KnownPerson(user_id=uuid5(NAMESPACE_URL, name), name=name)


def _unit(*values: float) -> Embedding:
    vector = np.array(values, np.float32)
    return vector / np.linalg.norm(vector)


def _detection(x: float, embedding_id: float) -> Detection:
    # `embedding_id` rides on the score so that a fake embedder can tell faces apart.
    return Detection(
        box=BoundingBox(x=x, y=100.0, width=100.0, height=120.0),
        landmarks=(
            (x + 30, 140.0),
            (x + 70, 140.0),
            (x + 50, 165.0),
            (x + 35, 190.0),
            (x + 65, 190.0),
        ),
        score=0.9 + embedding_id,
    )


def _embed_by_score(detection: Detection) -> Embedding:
    return {0.9: _unit(1, 0, 0), 0.95: _unit(0, 1, 0)}[round(detection.score, 2)]


def _yunet_outputs(height: int, width: int) -> dict[str, np.ndarray]:
    outputs = {}
    for stride in (8, 16, 32):
        anchors = (height // stride) * (width // stride)
        outputs[f"cls_{stride}"] = np.zeros((1, anchors, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, anchors, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, anchors, 4), np.float32)
        outputs[f"kps_{stride}"] = np.zeros((1, anchors, 10), np.float32)
    return outputs


def test_decode_yunet_places_a_face_from_its_anchor() -> None:
    outputs = _yunet_outputs(64, 64)
    # Stride 8 on a 64-wide input: 8 anchors per row, so anchor 9 is row 1, column 1.
    outputs["cls_8"][0, 9, 0] = 0.81
    outputs["obj_8"][0, 9, 0] = 1.0
    outputs["bbox_8"][0, 9] = [0.5, 0.5, math.log(2), math.log(2)]
    outputs["kps_8"][0, 9] = [0, 0, 1, 0, 0.5, 0.5, 0, 1, 1, 1]
    # The same face, weaker, at stride 16: removed by non-maximum suppression.
    outputs["cls_16"][0, 0, 0] = 0.64
    outputs["obj_16"][0, 0, 0] = 1.0
    outputs["bbox_16"][0, 0] = [0.75, 0.75, 0.0, 0.0]

    (detection,) = decode_yunet(outputs, 64, score_threshold=0.5, nms_threshold=0.3)

    assert detection.box == BoundingBox(x=4.0, y=4.0, width=16.0, height=16.0)
    assert detection.landmarks == ((8.0, 8.0), (16.0, 8.0), (12.0, 12.0), (8.0, 16.0), (16.0, 16.0))
    assert detection.score == pytest.approx(0.9)


def test_align_maps_template_landmarks_to_the_same_crop() -> None:
    frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    template = (
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    )
    detection = Detection(
        box=BoundingBox(x=0, y=0, width=112, height=112), landmarks=template, score=1.0
    )

    assert np.array_equal(align(frame, detection), frame[:112, :112])


def test_gallery_recognizes_the_closest_person_above_the_threshold() -> None:
    gallery = Gallery(
        [
            (_person("Alice"), np.stack([_unit(1, 0, 0)])),
            (_person("Bob"), np.stack([_unit(0, 1, 0)])),
        ],
        MATCHING,
    )

    match = gallery.match([_unit(1, 0.2, 0)])
    assert match is not None
    assert (match.person, round(match.score, 2)) == (_person("Alice"), 0.98)
    assert gallery.match([_unit(0, 0, 1)]) is None


def test_gallery_calls_an_ambiguous_face_unknown() -> None:
    gallery = Gallery(
        [
            (_person("Alice"), np.stack([_unit(1, 0, 0)])),
            (_person("Bob"), np.stack([_unit(0, 1, 0)])),
        ],
        MATCHING,
    )

    assert gallery.match([_unit(1, 1, 0)]) is None


def test_gallery_gives_a_person_to_one_face_only() -> None:
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)

    assignments = gallery.assign({1: [_unit(1, 0.5, 0)], 2: [_unit(1, 0.1, 0)]})

    assert assignments[1] is None
    match = assignments[2]
    assert match is not None
    assert (match.person, round(match.score, 3)) == (_person("Alice"), 0.995)


def test_database_adds_trims_and_forgets(tmp_path: Path) -> None:
    database = FaceDatabase(tmp_path / "faces.sqlite3", max_embeddings_per_person=2)

    alice = database.add("Alice", [_unit(1, 0, 0), _unit(0, 1, 0)])
    assert database.add("Alice", [_unit(0, 0, 1)]) == alice
    database.add("Bob", [_unit(1, 1, 0)])

    loaded = FaceDatabase(tmp_path / "faces.sqlite3", max_embeddings_per_person=2).load()
    assert [person for person, _ in loaded] == [
        alice,
        KnownPerson(user_id=loaded[1][0].user_id, name="Bob"),
    ]
    assert np.allclose(loaded[0][1], np.stack([_unit(0, 1, 0), _unit(0, 0, 1)]))
    assert database.forget("Alice") == alice
    assert [person.name for person, _ in database.load()] == ["Bob"]
    with pytest.raises(UnknownPersonError):
        database.forget("Alice")


def test_database_keeps_names_apart_from_users_and_faces(tmp_path: Path) -> None:
    path = tmp_path / "faces.sqlite3"
    database = FaceDatabase(path, max_embeddings_per_person=2)
    alice = database.add("Alice", [_unit(1, 0, 0)])

    def rows(table: str) -> list[tuple[object, ...]]:
        with closing(sqlite3.connect(path)) as connection:
            return [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]

    assert [row[0] for row in rows("users")] == [str(alice.user_id)]
    assert rows("names") == [(str(alice.user_id), "Alice")]
    assert [row[1] for row in rows("embeddings")] == [str(alice.user_id)]
    assert database.people() == [alice]

    database.forget("Alice")

    assert (rows("users"), rows("names"), rows("embeddings")) == ([], [], [])


def test_tracker_announces_a_face_seen_long_enough_then_its_departure() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)

    assert tracker.update(0.0, [_detection(100, 0.0)], _embed_by_score, gallery) == []
    assert tracker.update(0.6, [_detection(110, 0.0)], _embed_by_score, gallery) == [
        FaceAppeared(
            face=TrackedFace(
                track_id=1,
                box=BoundingBox(x=110.0, y=100.0, width=100.0, height=120.0),
                person=_person("Alice"),
                nearest=True,
            )
        )
    ]
    assert tracker.update(2.0, [], _embed_by_score, gallery) == []
    assert tracker.update(3.6, [], _embed_by_score, gallery) == [
        FaceLeft(track_id=1, person=_person("Alice"))
    ]


def test_tracker_drops_a_new_face_missed_before_it_is_announced() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([], MATCHING)

    tracker.update(0.0, [_detection(100, 0.0)], _embed_by_score, gallery)
    tracker.update(0.1, [], _embed_by_score, gallery)

    assert tracker.update(0.6, [_detection(100, 0.0)], _embed_by_score, gallery) == []
    assert [track.track_id for track in tracker.tracks] == [2]


def test_tracker_keeps_the_track_of_a_face_that_jumped_by_its_embedding() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([], MATCHING)
    tracker.update(0.0, [_detection(100, 0.0), _detection(1000, 0.05)], _embed_by_score, gallery)
    tracker.update(0.6, [_detection(100, 0.0), _detection(1000, 0.05)], _embed_by_score, gallery)

    # The second face moved far away in one frame: no overlap, same embedding.
    tracker.update(0.7, [_detection(100, 0.0), _detection(500, 0.05)], _embed_by_score, gallery)

    # Both faces are the same size: the first one counts as the nearest.
    assert [(face.track_id, face.box.x, face.nearest) for face in tracker.faces()] == [
        (1, 100.0, True),
        (2, 500.0, False),
    ]


def test_window_label_shows_track_name_and_user_id() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)
    tracker.update(0.0, [_detection(100, 0.0), _detection(400, 0.05)], _embed_by_score, gallery)

    assert [label(track) for track in tracker.tracks] == [
        f"#1 Alice ({str(_person('Alice').user_id)[:8]}) 1.00",
        "#2 unknown",
    ]


def test_database_renames_and_grants_without_changing_the_user(tmp_path: Path) -> None:
    database = FaceDatabase(tmp_path / "faces.sqlite3", max_embeddings_per_person=2)
    alice = database.add("Alice", [_unit(1, 0, 0)])
    bob = database.add("Bob", [_unit(0, 1, 0)])

    database.grant_operator(bob.user_id, granted_by=alice.user_id)
    renamed = database.rename(bob.user_id, "Robert")

    assert renamed == KnownPerson(user_id=bob.user_id, name="Robert")
    assert database.people() == [alice, renamed]
    assert database.granted_operators() == frozenset({bob.user_id})
    with pytest.raises(NameTakenError):
        database.rename(bob.user_id, "Alice")
    assert database.revoke_operator(bob.user_id)
    assert not database.revoke_operator(bob.user_id)


def test_face_boxes_are_scaled_and_carry_what_the_speaker_said() -> None:
    speaker = Track(
        track_id=3, box=BoundingBox(x=100, y=200, width=50, height=80), now=0, samples=1
    )
    other = Track(track_id=4, box=BoundingBox(x=0, y=0, width=10, height=10), now=0, samples=1)

    boxes = face_boxes([speaker, other], scale=0.5, width=960, height=540, caption=(3, "hello"))

    assert boxes == FaceBoxes(
        image_width=960,
        image_height=540,
        boxes=(
            FaceBox(
                track_id=3,
                box=BoundingBox(x=50, y=100, width=25, height=40),
                person=None,
                confirmed=False,
                said="hello",
            ),
            FaceBox(
                track_id=4,
                box=BoundingBox(x=0, y=0, width=5, height=5),
                person=None,
                confirmed=False,
                said=None,
            ),
        ),
    )
