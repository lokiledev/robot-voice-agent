import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Final, assert_never

from loguru import logger

from rva_interaction.config import InteractionConfig
from rva_interaction.engagement import EngagementTracker
from rva_interaction.gaze import GazeHistory
from rva_interaction.speakers import SpeakerAttribution
from rva_schemas.abc.recorder import Recorder
from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, UserUtterance
from rva_schemas.interaction import EngagementEvent, EngagementState
from rva_schemas.observation import VisualObservation


class Interaction:
    """`Engagement` from what the camera sees, and who each utterance is from and to.

    Where perception meets the dialogue: who pays attention to the robot, which face said an
    utterance, and whether they said it to the robot.
    """

    def __init__(
        self, config: InteractionConfig, score_delay_s: float | None, recorder: Recorder
    ) -> None:
        """Start with nobody seen.

        Args:
            config: Thresholds and time constants.
            score_delay_s: How long before its frame the moment the active speaker scores is;
                `None` without an active speaker, when no utterance can be attributed.
            recorder: Receives the engagement of each frame.
        """
        self._config: Final = config
        self._recorder: Final = recorder
        self._engagement: Final = EngagementTracker(config.engagement, config.eye_contact_deg)
        self._gaze: Final = GazeHistory(config.history_s, config.eye_contact_deg)
        self._speakers: Final = (
            None
            if score_delay_s is None
            else SpeakerAttribution(config.history_s, config.min_speaking_share, score_delay_s)
        )
        self._speaking: frozenset[int] = frozenset()
        """The tracks seen speaking in the last frame."""
        self._subscribers: Final[set[asyncio.Queue[EngagementEvent]]] = set()

    async def run(self, observations: AsyncIterator[VisualObservation]) -> None:
        """Follow what the camera sees until `observations` ends."""
        async for observation in observations:
            self.observe(observation)

    def observe(self, observation: VisualObservation) -> EngagementState:
        self._gaze.observe(observation)
        if self._speakers is not None:
            self._speakers.observe(observation)
        self._speaking = frozenset(
            person.track_id
            for person in observation.people.people
            if person.speaker is not None and person.speaker.speaking
        )
        state = self._engagement.observe(observation)
        self._recorder.record(state)
        for event in state.events:
            logger.info("{}", event)
            for queue in self._subscribers:
                queue.put_nowait(event)
        return state

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[EngagementEvent]]:
        queue = asyncio.Queue[EngagementEvent]()
        self._subscribers.add(queue)
        try:
            yield _drain(queue)
        finally:
            self._subscribers.discard(queue)

    def engaged(self) -> frozenset[int]:
        return self._engagement.engaged

    def may_interrupt(self) -> bool:
        """Whether speech heard now may stop the robot's: see `InteractionConfig.barge_in`."""
        match self._config.barge_in:
            case "anyone":
                return True
            case "engaged":
                engaged = self.engaged()
                return not engaged or bool(engaged & self._speaking)
            case _:
                assert_never(self._config.barge_in)

    def hear(self, utterance: UserUtterance) -> UserUtterance:
        """`utterance` with who said it and whether to the robot, if it was spoken.

        A typed line passes as it is, and so does everything without an active speaker. Someone
        not engaged who speaks to the robot is engaged from then on.
        """
        if self._speakers is None or utterance.speech is None:
            return utterance
        speaker = self._speakers.attribute(utterance.speech)
        match speaker:
            case NobodySeenSpeaking():
                pass
            case SeenSpeaking():
                engaged = speaker.track_id in self.engaged()
                gaze = self._gaze.share(speaker.track_id, utterance.speech)
                needed = (
                    self._config.engaged_min_gaze if engaged else self._config.bystander_min_gaze
                )
                # A speaker whose face was not seen cannot be told to look elsewhere: trusted if
                # engaged, not otherwise.
                to_robot = engaged if gaze is None else gaze >= needed
                logger.debug(
                    "Track {} looked at the robot for {} of it, engaged: {}",
                    speaker.track_id,
                    gaze,
                    engaged,
                )
                if to_robot:
                    self._engagement.engage(speaker.track_id)
                speaker = speaker.model_copy(update={"to_robot": to_robot})
            case _:
                assert_never(speaker)
        logger.info("{!r} said by {}", utterance.text, speaker)
        return utterance.model_copy(update={"speaker": speaker})


async def _drain(queue: asyncio.Queue[EngagementEvent]) -> AsyncIterator[EngagementEvent]:
    while True:
        yield await queue.get()
