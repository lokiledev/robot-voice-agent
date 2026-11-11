from collections import Counter, deque
from typing import Final, NamedTuple

from loguru import logger

from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan, UtteranceSpeaker
from rva_schemas.observation import VisualObservation
from rva_schemas.vision import KnownPerson


class SpeakerAttribution:
    """Tells which face in view said an utterance, from who was seen speaking while it was heard.

    The active speaker often sees a listener speak too, nodding or smiling along, so in each frame
    only the face it is surest of counts. Fed one observation per frame, in order. It runs on the
    frames' times and does no I/O: fed a recorded sequence, it attributes the same.
    """

    def __init__(
        self, history_s: float, min_share: float, min_segment_s: float, score_delay_s: float
    ) -> None:
        """Start with nothing seen.

        Args:
            history_s: How far back who was speaking is kept.
            min_share: Share of an utterance's frames a face must be the one speaking in.
            min_segment_s: The shortest piece a turn is cut into where the speaker changes.
            score_delay_s: How long before its frame the moment the active speaker scores is.
        """
        self._history_s: Final = history_s
        self._min_share: Final = min_share
        self._min_segment_s: Final = min_segment_s
        self._score_delay_s: Final = score_delay_s
        self._frames: Final = deque[_Frame]()

    def observe(self, observation: VisualObservation) -> None:
        time = observation.time - self._score_delay_s
        speaking = [
            (person.speaker.score, person)
            for person in observation.people.people
            if person.confirmed and person.speaker is not None and person.speaker.speaking
        ]
        _, surest = max(speaking, key=lambda scored: scored[0], default=(None, None))
        self._frames.append(
            _Frame(
                time,
                None if surest is None else surest.track_id,
                None if surest is None else surest.person,
            )
        )
        while self._frames[0].time < time - self._history_s:
            self._frames.popleft()

    def segments(self, speech: SpeechSpan) -> tuple[SpeechSpan, ...]:
        """`speech` cut where the face speaking changes, into pieces that cover it in order.

        A stretch shorter than `min_segment_s` joins the piece before it, and the frames where
        nobody is seen speaking, the pauses, join whoever spoke before them. One piece when one
        face spoke throughout, or none was seen to.
        """
        frames = self._during(speech)
        # Each run of frames with the same speaker, as the speaker and when it starts.
        runs: list[tuple[int, float]] = []
        for frame in frames:
            if frame.track_id is not None and (not runs or runs[-1][0] != frame.track_id):
                runs.append((frame.track_id, frame.time))
        while len(runs) > 1:
            ends = [start for _, start in runs[1:]] + [speech.end]
            lengths = [end - start for (_, start), end in zip(runs, ends, strict=True)]
            shortest = min(range(len(runs)), key=lengths.__getitem__)
            if lengths[shortest] >= self._min_segment_s:
                break
            if shortest == 0:
                # Nothing before it to join: the next piece starts where it did.
                runs[1] = (runs[1][0], runs[0][1])
            del runs[shortest]
            # The pieces on either side of the one removed may be the same speaker's.
            runs = [run for i, run in enumerate(runs) if i == 0 or run[0] != runs[i - 1][0]]
        if len(runs) <= 1:
            return (speech,)
        cuts = [speech.start, *(start for _, start in runs[1:]), speech.end]
        return tuple(SpeechSpan(start=a, end=b) for a, b in zip(cuts, cuts[1:], strict=False))

    def attribute(self, speech: SpeechSpan) -> UtteranceSpeaker:
        """Who said what was heard over `speech`: the face seen speaking for most of it.

        Nobody, when no face was the one speaking for `min_share` of it: the speaker is out of
        view, or turned away. Whether it was said to the robot is not judged here: it is taken
        to be.
        """
        frames = self._during(speech)
        seen = Counter(frame.track_id for frame in frames if frame.track_id is not None)
        shares = {track_id: count / len(frames) for track_id, count in seen.items()}
        logger.debug("Share of {} frames each track spoke in: {}", len(frames), shares)
        if not shares:
            return NobodySeenSpeaking()
        track_id, share = max(shares.items(), key=lambda item: item[1])
        if share < self._min_share:
            return NobodySeenSpeaking()
        # Whoever the face was last recognized as: a name can come in the middle of a sentence.
        person = next(frame.person for frame in reversed(frames) if frame.track_id == track_id)
        return SeenSpeaking(track_id=track_id, person=person, to_robot=True)

    def _during(self, speech: SpeechSpan) -> list["_Frame"]:
        return [frame for frame in self._frames if speech.start <= frame.time <= speech.end]


class _Frame(NamedTuple):
    time: float
    """The moment the active speaker scored, earlier than the frame it came with."""
    track_id: int | None
    """The face seen speaking then, the surest when several were; `None` when none was."""
    person: KnownPerson | None
    """Who that face was recognized as."""
