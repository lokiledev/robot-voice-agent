import asyncio
from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import pytest

from rva_app.gaze import GazeOut
from rva_robot_client.head import gaze_targets
from rva_robot_schemas.head import GazeTarget
from rva_schemas.attention import AttentionState, AttentionTarget, FaceEntity
from rva_schemas.vision import Point3D


@pytest.fixture
def namespace() -> Iterator[str]:
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


def _state(time: float, x: float) -> AttentionState:
    target = AttentionTarget(
        entity=FaceEntity(track_id=3, person=None), point=Point3D(x=x, y=-0.1, z=1.2)
    )
    return AttentionState(time=time, target=target, candidates=())


async def test_targets_go_to_the_controller_once_it_runs(namespace: str) -> None:
    gaze = GazeOut(namespace)
    # Nobody to send to yet: dropped.
    gaze.send(_state(10.0, 0.1))

    async with gaze_targets(namespace) as targets:
        gaze.send(_state(10.1, 0.2))
        gaze.send(AttentionState(time=10.2, target=None, candidates=()))

        received = await asyncio.wait_for(anext(targets), timeout=5)

    assert received == GazeTarget(timestamp=10.1, x=0.2, y=-0.1, z=1.2)
