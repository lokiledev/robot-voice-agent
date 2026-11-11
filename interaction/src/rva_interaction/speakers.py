from collections import Counter, deque
from typing import Final, NamedTuple

from loguru import logger

from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan, UtteranceSpeaker
from rva_schemas.observation import VisualObservation
from rva_schemas.vision import KnownPerson


class SpeakerAttribution:
    """Tells which face in view said an utterance, from who was seen speaking while it was heard.

    Fed one observation per frame, in order. It runs on the frames' times and does no I/O: fed a
    recorded sequence, it attributes the same.
    """

    def __init__(self, history_s: float, min_share: float, score_delay_s: float) -> None:
        """Start with nothing seen.

        Args:
            history_s: How far back who was speaking is kept.
            min_share: Share of an utterance's frames a face must be seen speaking in.
            score_delay_s: How long before its frame the moment the active speaker scores is.
        """
        self._history_s: Final = history_s
        self._min_share: Final = min_share
        self._score_delay_s: Final = score_delay_s
        self._frames: Final = deque[_Frame]()

    def observe(self, observation: VisualObservation) -> None:
        time = observation.time - self._score_delay_s
        self._frames.append(
            _Frame(
                time,
                {
                    person.track_id: person.person
                    for person in observation.people.people
                    if person.confirmed and person.speaker is not None and person.speaker.speaking
                },
            )
        )
        while self._frames[0].time < time - self._history_s:
            self._frames.popleft()

    def attribute(self, speech: SpeechSpan) -> UtteranceSpeaker:
        """Who said what was heard over `speech`: the face seen speaking for most of it.

        Nobody, when no face was seen speaking for `min_share` of it: the speaker is out of
        view, or turned away. Whether it was said to the robot is not judged here: it is taken
        to be.
        """
        frames = [frame for frame in self._frames if speech.start <= frame.time <= speech.end]
        seen = Counter(track_id for frame in frames for track_id in frame.speaking)
        shares = {track_id: count / len(frames) for track_id, count in seen.items()}
        logger.debug("Share of {} frames each track spoke in: {}", len(frames), shares)
        if not shares:
            return NobodySeenSpeaking()
        track_id, share = max(shares.items(), key=lambda item: item[1])
        if share < self._min_share:
            return NobodySeenSpeaking()
        # Whoever the face was last recognized as: a name can come in the middle of a sentence.
        person = next(
            frame.speaking[track_id] for frame in reversed(frames) if track_id in frame.speaking
        )
        return SeenSpeaking(track_id=track_id, person=person, to_robot=True)


class _Frame(NamedTuple):
    time: float
    """The moment the active speaker scored, earlier than the frame it came with."""
    speaking: dict[int, KnownPerson | None]
    """Each face seen speaking then, by track id, with who it was recognized as."""
