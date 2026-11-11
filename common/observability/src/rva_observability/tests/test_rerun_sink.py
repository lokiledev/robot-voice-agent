from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from rva_observability.rerun_sink import RerunRecorder
from rva_schemas.audio import AudioClip
from rva_schemas.record import (
    CameraFrame,
    DepthFrame,
    PeopleInView,
    Sound,
    Speaker,
    TrackedPerson,
    UserLine,
)
from rva_schemas.vision import (
    ActiveSpeaker,
    Body,
    BoundingBox,
    CameraIntrinsics,
    DepthMap,
    Face3D,
    FaceFeatures,
    Gaze,
    GazeRay,
    Hand,
    Keypoint,
    KnownPerson,
    Point3D,
)


def test_people_are_drawn_on_the_image_and_in_3d(tmp_path: Path) -> None:
    path = tmp_path / "session.rrd"
    recording = rr.RecordingStream("test")
    recording.save(path)
    recorder = RerunRecorder(recording)
    _, jpeg = cv2.imencode(".jpg", np.zeros((54, 96, 3), dtype=np.uint8))
    alice = KnownPerson(user_id=UUID(int=1), name="Alice")

    intrinsics = CameraIntrinsics(
        calibrated=False, width=96, height=54, fx=68.0, fy=68.0, cx=47.5, cy=26.5
    )

    recorder.record(CameraFrame(jpeg=jpeg.tobytes()))
    recorder.record(
        DepthFrame(
            depth=DepthMap(metres=np.full((27, 48), 1.5, np.float32), calibrated=False),
            intrinsics=intrinsics.scaled_to(48, 27),
        )
    )
    # 21 points: the wrist and the thumb found, the other fingers not.
    hand = Hand(
        keypoints=(
            *(
                Keypoint(x=10, y=20 + i, score=0.9, position=Point3D(x=0.1, y=i / 100, z=1.5))
                for i in range(5)
            ),
            *(None,) * 16,
        )
    )
    body = Body(
        box=BoundingBox(x=0, y=0, width=50, height=54),
        keypoints=(
            Keypoint(x=2, y=4, score=0.8, position=Point3D(x=-0.5, y=-0.3, z=1.5)),
            None,
            Keypoint(x=3, y=4, score=0.7, position=None),
        ),
        left_hand=hand,
        right_hand=hand,
    )
    face = FaceFeatures(
        landmarks=tuple((float(i % 10), float(i % 7)) for i in range(478)),
        left_eye_open=0.9,
        right_eye_open=0.8,
        mouth_open=0.3,
        gaze=Gaze(yaw=5.0, pitch=-2.0),
        in_3d=Face3D(
            landmarks=tuple(Point3D(x=i / 1000, y=-0.2, z=1.5) for i in range(478)),
            gaze=GazeRay(
                origin=Point3D(x=0.0, y=-0.2, z=1.5), direction=Point3D(x=0.0, y=0.0, z=-1.0)
            ),
        ),
        expression="happy",
        expression_score=0.7,
    )
    recorder.record(
        PeopleInView(
            image_width=96,
            image_height=54,
            intrinsics=intrinsics,
            people=(
                TrackedPerson(
                    track_id=1,
                    box=BoundingBox(x=1, y=2, width=3, height=4),
                    person=alice,
                    match_score=0.8,
                    confirmed=True,
                    said="authorize Bob",
                    face=face,
                    body=body,
                    speaker=ActiveSpeaker(score=0.7, speaking=True),
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
        "/camera",
        "/camera/image",
        "/depth",
        "/depth/image",
        "/space/people/1/face",
        "/space/people/1/gaze",
        "/space/people/1/body",
        "/space/people/1/body/left_hand",
        "/space/unmatched_bodies/0",
        "/camera/people/1/box",
        "/camera/people/1/face",
        "/camera/people/1/gaze",
        "/camera/people/1/body",
        "/camera/people/1/body/left_hand",
        "/camera/unmatched_bodies/0",
        "/people/1/mouth",
        "/people/1/speaking",
        "/dialogue/user",
    } <= entities


def test_sound_is_logged_as_clips_and_as_its_level_over_time(tmp_path: Path) -> None:
    path = tmp_path / "session.rrd"
    recording = rr.RecordingStream("test")
    recording.save(path)
    recorder = RerunRecorder(recording)
    frame = AudioClip(samples=np.full(512, 0.5, np.float32), sample_rate=16000)

    # A second and a half of the microphone, in frames of 32 ms: one clip of a second.
    for _ in range(47):
        recorder.record(Sound(source="microphone", clip=frame))
    # Half a second of the robot's voice: one clip.
    recorder.record(
        Sound(
            source="robot", clip=AudioClip(samples=np.zeros(12000, np.float32), sample_rate=24000)
        )
    )
    recording.flush()
    recording.disconnect()

    rows: dict[str, int] = {}
    for chunk in RrdReader(path).stream():
        rows[chunk.entity_path] = rows.get(chunk.entity_path, 0) + chunk.num_rows
    assert {name: count for name, count in rows.items() if name.startswith("/audio")} == {
        "/audio/microphone": 1,
        "/audio/microphone/level": 47,
        "/audio/robot": 1,
        # 0.5 s in steps of 32 ms.
        "/audio/robot/level": 16,
    }
