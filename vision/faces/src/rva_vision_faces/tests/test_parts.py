import math
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import (
    Body,
    BoundingBox,
    FaceAppeared,
    FaceFeatures,
    FaceLeft,
    Gaze,
    Hand,
    Keypoint,
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
from rva_vision_faces.overlay import label
from rva_vision_faces.scene import Scene, people_in_view
from rva_vision_faces.tracker import Track, Tracker

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


def test_tracker_stops_embedding_a_recognized_track_and_keeps_its_person() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)
    embedded: list[float] = []

    def embed(detection: Detection) -> Embedding:
        embedded.append(detection.box.x)
        return _embed_by_score(detection)

    tracker.update(0.0, [_detection(100, 0.0)], embed, gallery)
    # Seconds later, with samples due, the face moved a little: still Alice, not embedded again.
    tracker.update(1.5, [_detection(110, 0.0)], embed, gallery)
    tracker.update(3.0, [_detection(120, 0.0)], embed, gallery)

    assert embedded == [100]
    assert [face.person for face in tracker.faces()] == [_person("Alice")]


def test_tracker_samples_an_unknown_track_until_it_is_recognized() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)
    samples = iter([_unit(0, 0, 1), _unit(0, 0, 1), _unit(1, 0.1, 0), _unit(0, 0, 1)])

    def embed(detection: Detection) -> Embedding:
        return next(samples)

    tracker.update(0.0, [_detection(100, 0.0)], embed, gallery)
    tracker.update(0.5, [_detection(100, 0.0)], embed, gallery)
    assert [face.person for face in tracker.faces()] == [None]
    tracker.update(1.0, [_detection(100, 0.0)], embed, gallery)
    assert [face.person for face in tracker.faces()] == [None]
    tracker.update(2.0, [_detection(100, 0.0)], embed, gallery)
    assert [face.person for face in tracker.faces()] == [_person("Alice")]
    tracker.update(3.0, [_detection(100, 0.0)], embed, gallery)
    assert [face.person for face in tracker.faces()] == [_person("Alice")]


def test_tracker_gives_a_person_already_on_a_track_to_no_other() -> None:
    tracker = Tracker(TRACKING)
    gallery = Gallery([(_person("Alice"), np.stack([_unit(1, 0, 0)]))], MATCHING)
    tracker.update(0.0, [_detection(100, 0.0)], lambda _: _unit(1, 0.5, 0), gallery)

    # A second face, closer to Alice's samples, appears while the first is still followed.
    tracker.update(
        0.6, [_detection(100, 0.0), _detection(1000, 0.05)], lambda _: _unit(1, 0, 0), gallery
    )

    assert [
        (track.track_id, None if track.match is None else track.match.person)
        for track in tracker.tracks
    ] == [(1, _person("Alice")), (2, None)]


def test_tracker_recognizes_every_track_again_when_people_change() -> None:
    tracker = Tracker(TRACKING)
    alice = _person("Alice")
    tracker.update(
        0.0,
        [_detection(100, 0.0)],
        _embed_by_score,
        Gallery([(alice, np.stack([_unit(1, 0, 0)]))], MATCHING),
    )
    renamed = KnownPerson(user_id=alice.user_id, name="Alicia")

    tracker.recognize(Gallery([(renamed, np.stack([_unit(1, 0, 0)]))], MATCHING))
    assert [track.match.person for track in tracker.tracks if track.match is not None] == [renamed]
    tracker.recognize(Gallery([], MATCHING))
    assert [track.match for track in tracker.tracks] == [None]


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


def test_people_in_view_are_scaled_and_carry_what_the_speaker_said() -> None:
    speaker = Track(track_id=3, detection=_detection(100, 0.0), now=0, samples=1)
    other = Track(track_id=4, detection=_detection(400, 0.0), now=0, samples=1)
    hand = Hand(keypoints=(Keypoint(x=10, y=20, score=0.9), None))
    body = Body(
        box=BoundingBox(x=100, y=100, width=200, height=400),
        keypoints=(Keypoint(x=150, y=160, score=0.8), None),
        left_hand=hand,
        right_hand=hand,
    )
    face = FaceFeatures(
        landmarks=((120.0, 140.0), (180.0, 140.0)),
        left_eye_open=0.9,
        right_eye_open=0.8,
        mouth_open=0.1,
        gaze=Gaze(yaw=5.0, pitch=-2.0),
        expression="happy",
        expression_score=0.7,
    )
    scene = Scene(faces={3: face}, bodies={3: body}, unmatched_bodies=(body,), speakers={})

    people = people_in_view(
        [speaker, other], scene, scale=0.5, width=960, height=540, caption=(3, "hello")
    )

    half_hand = Hand(keypoints=(Keypoint(x=5, y=10, score=0.9), None))
    half_body = Body(
        box=BoundingBox(x=50, y=50, width=100, height=200),
        keypoints=(Keypoint(x=75, y=80, score=0.8), None),
        left_hand=half_hand,
        right_hand=half_hand,
    )
    assert people == PeopleInView(
        image_width=960,
        image_height=540,
        people=(
            TrackedPerson(
                track_id=3,
                box=BoundingBox(x=50, y=50, width=50, height=60),
                person=None,
                confirmed=False,
                said="hello",
                face=face.model_copy(update={"landmarks": ((60.0, 70.0), (90.0, 70.0))}),
                body=half_body,
                speaker=None,
            ),
            TrackedPerson(
                track_id=4,
                box=BoundingBox(x=200, y=50, width=50, height=60),
                person=None,
                confirmed=False,
                said=None,
                face=None,
                body=None,
                speaker=None,
            ),
        ),
        unmatched_bodies=(half_body,),
    )
