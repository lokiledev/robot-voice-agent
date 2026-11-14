from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import pytest

from rva_app.head_camera import PublishedCameraPoses
from rva_robot_client.head import CameraPoseReader, advertise_camera_poses
from rva_robot_schemas.head import CameraPoseStamped
from rva_schemas.frames import CameraPose

AHEAD = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
"""The optical frame looking along the base's x: its z ahead, x to the right, y down."""


@pytest.fixture
def namespace() -> Iterator[str]:
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


def test_a_frame_gets_the_camera_pose_nearest_its_time_within_the_gap(namespace: str) -> None:
    poses = PublishedCameraPoses(CameraPoseReader(namespace, "test_vision"))
    with advertise_camera_poses(namespace) as publisher:
        assert poses.at(10.0) is None, "subscribed, nothing published yet"
        for i, x in enumerate((0.0, 0.1, 0.2)):
            publisher.publish(
                CameraPoseStamped(timestamp=10.0 + i / 100, position=(x, 0.0, 0.2), rotation=AHEAD)
            )

        nearest = poses.at(10.012)
        too_late = poses.at(10.1)

    assert nearest == CameraPose(position=(0.1, 0.0, 0.2), rotation=AHEAD)
    assert too_late is None
