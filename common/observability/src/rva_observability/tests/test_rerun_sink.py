from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from rva_observability.rerun_sink import RerunRecorder
from rva_schemas.record import CameraFrame, PeopleInView, Speaker, TrackedPerson, UserLine
from rva_schemas.vision import Body, BoundingBox, FaceFeatures, Gaze, Hand, Keypoint, KnownPerson


def test_camera_image_and_people_share_one_view(tmp_path: Path) -> None:
    path = tmp_path / "session.rrd"
    recording = rr.RecordingStream("test")
    recording.save(path)
    recorder = RerunRecorder(recording)
    _, jpeg = cv2.imencode(".jpg", np.zeros((54, 96, 3), dtype=np.uint8))
    alice = KnownPerson(user_id=UUID(int=1), name="Alice")

    recorder.record(CameraFrame(jpeg=jpeg.tobytes()))
    # 21 points: the wrist and the thumb found, the other fingers not.
    hand = Hand(keypoints=(*(Keypoint(x=10, y=20 + i, score=0.9) for i in range(5)), *(None,) * 16))
    body = Body(
        box=BoundingBox(x=0, y=0, width=50, height=54),
        keypoints=(Keypoint(x=2, y=4, score=0.8), None, Keypoint(x=3, y=4, score=0.7)),
        left_hand=hand,
        right_hand=hand,
    )
    face = FaceFeatures(
        landmarks=tuple((float(i % 10), float(i % 7)) for i in range(478)),
        left_eye_open=0.9,
        right_eye_open=0.8,
        mouth_open=0.3,
        gaze=Gaze(yaw=5.0, pitch=-2.0),
        expression="happy",
        expression_score=0.7,
    )
    recorder.record(
        PeopleInView(
            image_width=96,
            image_height=54,
            people=(
                TrackedPerson(
                    track_id=1,
                    box=BoundingBox(x=1, y=2, width=3, height=4),
                    person=alice,
                    confirmed=True,
                    said="authorize Bob",
                    face=face,
                    body=body,
                ),
            ),
            unmatched_bodies=(body,),
        )
    )
    recorder.record(
        UserLine(text="authorize Bob", speaker=Speaker(track_id=1, person=alice, operator=True))
    )
    recording.flush()
    recording.disconnect()

    entities = {chunk.entity_path for chunk in RrdReader(path).stream()}
    assert {
        "/camera/image",
        "/camera/people/1/box",
        "/camera/people/1/face",
        "/camera/people/1/gaze",
        "/camera/people/1/body",
        "/camera/people/1/body/left_hand",
        "/camera/unmatched_bodies/0",
        "/people/1/mouth",
        "/dialogue/user",
    } <= entities
