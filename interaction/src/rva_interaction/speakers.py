from collections import Counter, deque
from collections.abc import Sequence
from typing import Final, NamedTuple

from loguru import logger

from rva_interaction.bearings import apart, face_bearing, speech_bearing, view_bearings
from rva_interaction.config import SoundDirectionConfig
from rva_schemas.dialogue import NobodySeenSpeaking, SeenSpeaking, SpeechSpan, UtteranceSpeaker
from rva_schemas.frames import StaticTransforms
from rva_schemas.observation import VisualObservation
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.sound import HeardDirection
from rva_schemas.vision import KnownPerson


class SpeakerAttribution:
    """Tells which face in view said an utterance, from who was seen speaking while it was heard.

    One face counts as speaking in each frame. The active speaker often sees a listener speak too,
    nodding or smiling along, so it is the one it is surest of; but with a microphone array, the
    direction the speech comes from decides between them first. Fed one observation per frame,
    and the directions heard, each in order. It runs on their times and does no I/O: fed a
    recorded sequence, it attributes the same.
    """

    def __init__(
        self,
        history_s: float,
        min_share: float,
        min_segment_s: float,
        score_delay_s: float,
        sound: SoundDirectionConfig,
        transforms: StaticTransforms,
    ) -> None:
        """Start with nothing seen or heard.

        Args:
            history_s: How far back who was speaking is kept.
            min_share: Share of an utterance's frames a face must be the one speaking in.
            min_segment_s: The shortest piece a turn is cut into where the speaker changes.
            score_delay_s: How long before its frame the moment the active speaker scores is.
            sound: How a direction heard tells which face speaks.
            transforms: Turn a direction heard into the camera's frame.
        """
        self._history_s: Final = history_s
        self._min_share: Final = min_share
        self._min_segment_s: Final = min_segment_s
        self._score_delay_s: Final = score_delay_s
        self._sound: Final = sound
        self._transforms: Final = transforms
        self._frames: Final = deque[_Frame]()
        self._directions: Final = deque[HeardDirection]()

    def hear(self, direction: HeardDirection) -> None:
        """Take where a microphone array heard speech from, or that it heard none."""
        self._directions.append(direction)
        while self._directions[0].time < direction.time - self._history_s:
            self._directions.popleft()

    def observe(self, observation: VisualObservation) -> None:
        time = observation.time - self._score_delay_s
        speaker = self._speaking(observation.people, self._heard_at(time))
        self._frames.append(
            _Frame(
                time,
                None if speaker is None else speaker.track_id,
                None if speaker is None else speaker.person,
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

    def _speaking(self, people: PeopleInView, heard: HeardDirection | None) -> TrackedPerson | None:
        """The face speaking in a frame, if any, from what was seen and heard at its time."""
        confirmed = [person for person in people.people if person.confirmed]
        speaking = [
            (person.speaker.score, person)
            for person in confirmed
            if person.speaker is not None and person.speaker.speaking
        ]
        surest = _surest(speaking)
        intrinsics = people.intrinsics
        bearing = (
            None
            if heard is None or intrinsics is None
            else speech_bearing(heard, people.frame, self._transforms)
        )
        if bearing is None or intrinsics is None:
            return surest
        tolerance = self._sound.tolerance_deg
        toward = [
            person
            for person in confirmed
            if apart(face_bearing(person.box, intrinsics), bearing) <= tolerance
        ]
        heard_from = {person.track_id for person in toward}
        seen = _surest([scored for scored in speaking if scored[1].track_id in heard_from])
        if seen is not None:
            return seen
        left, right = view_bearings(intrinsics)
        if not left - tolerance <= bearing <= right + tolerance:
            # Someone out of view speaks: lips moving in view are a listener's.
            return None
        if surest is not None:
            # The speech comes from where no face is seen speaking: the direction is too rough
            # to overrule the lips.
            return surest
        # Turned away, or the lips hidden: heard from where only they are.
        return toward[0] if len(toward) == 1 else None

    def _heard_at(self, time: float) -> HeardDirection | None:
        """The last direction heard by `time`, unless older than it holds for."""
        for direction in reversed(self._directions):
            if direction.time <= time:
                return direction if time - direction.time <= self._sound.max_age_s else None
        return None

    def _during(self, speech: SpeechSpan) -> list["_Frame"]:
        return [frame for frame in self._frames if speech.start <= frame.time <= speech.end]


class _Frame(NamedTuple):
    time: float
    """The moment the active speaker scored, earlier than the frame it came with."""
    track_id: int | None
    """The face seen speaking then, the surest when several were; `None` when none was."""
    person: KnownPerson | None
    """Who that face was recognized as."""


def _surest(speaking: Sequence[tuple[float, TrackedPerson]]) -> TrackedPerson | None:
    """The face the active speaker is surest of among those it saw speaking."""
    _, surest = max(speaking, key=lambda scored: scored[0], default=(None, None))
    return surest
