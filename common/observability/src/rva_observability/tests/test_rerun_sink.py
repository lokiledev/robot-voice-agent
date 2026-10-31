from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from rva_observability.rerun_sink import RerunRecorder
from rva_schemas.record import CameraFrame, FaceBox, FaceBoxes, Speaker, UserLine
from rva_schemas.vision import BoundingBox, KnownPerson


def test_camera_image_and_faces_share_one_view(tmp_path: Path) -> None:
    path = tmp_path / "session.rrd"
    recording = rr.RecordingStream("test")
    recording.save(path)
    recorder = RerunRecorder(recording)
    _, jpeg = cv2.imencode(".jpg", np.zeros((54, 96, 3), dtype=np.uint8))
    alice = KnownPerson(user_id=UUID(int=1), name="Alice")

    recorder.record(CameraFrame(jpeg=jpeg.tobytes()))
    recorder.record(
        FaceBoxes(
            image_width=96,
            image_height=54,
            boxes=(
                FaceBox(
                    track_id=1,
                    box=BoundingBox(x=1, y=2, width=3, height=4),
                    person=alice,
                    confirmed=True,
                    said="authorize Bob",
                ),
            ),
        )
    )
    recorder.record(
        UserLine(text="authorize Bob", speaker=Speaker(track_id=1, person=alice, operator=True))
    )
    recording.flush()
    recording.disconnect()

    entities = {chunk.entity_path for chunk in RrdReader(path).stream()}
    assert {"/camera/image", "/camera/faces", "/dialogue/user"} <= entities
