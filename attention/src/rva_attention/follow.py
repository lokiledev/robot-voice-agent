from collections.abc import AsyncIterator
from typing import assert_never

from loguru import logger

from rva_attention.attention import VisualAttention
from rva_schemas.abc.recorder import Recorder
from rva_schemas.attention import AttentionEntity, AttentionState, FaceEntity, HandEntity
from rva_schemas.observation import VisualObservation


async def follow(
    observations: AsyncIterator[VisualObservation], attention: VisualAttention, recorder: Recorder
) -> AsyncIterator[AttentionState]:
    """Attend to each observation as it comes, recording each state and logging each new target.

    Yields every state, for whatever wants the target: the gaze, the body, a page.
    """
    previous: AttentionEntity | None = None
    async for observation in observations:
        state = attention.observe(observation)
        recorder.record(state)
        entity = None if state.target is None else state.target.entity
        if entity != previous:
            logger.info("Attending to {}", "nothing" if entity is None else describe(entity))
            previous = entity
        yield state


def describe(entity: AttentionEntity) -> str:
    match entity:
        case FaceEntity():
            who = "someone" if entity.person is None else entity.person.name
            return f"the face of {who} on track {entity.track_id}"
        case HandEntity():
            return f"the {entity.side} hand of track {entity.track_id}"
        case _:
            assert_never(entity)
