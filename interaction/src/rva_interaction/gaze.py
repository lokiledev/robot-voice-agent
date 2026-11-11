from collections import deque
from typing import Final, NamedTuple

from rva_schemas.dialogue import SpeechSpan
from rva_schemas.observation import VisualObservation


class GazeHistory:
    """Remembers who looked at the robot lately, to tell how much of an utterance its speaker did.

    Fed one observation per frame, in order; it runs on the frames' times and does no I/O.
    """

    def __init__(self, history_s: float, eye_contact_deg: float) -> None:
        self._history_s: Final = history_s
        self._eye_contact_deg: Final = eye_contact_deg
        self._frames: Final = deque[_Frame]()

    def observe(self, observation: VisualObservation) -> None:
        faces = {
            person.track_id: person.face.gaze.off_camera() <= self._eye_contact_deg
            for person in observation.people.people
            # Only a confirmed face is somebody: a detection of a frame or two is not.
            if person.face is not None and person.confirmed
        }
        self._frames.append(_Frame(observation.time, faces))
        while self._frames[0].time < observation.time - self._history_s:
            self._frames.popleft()

    def share(self, track_id: int, speech: SpeechSpan) -> float | None:
        """Share of the frames over `speech` in which the face on `track_id` looked at the robot.

        `None` if its face was not seen meanwhile.
        """
        looked = [
            frame.looking[track_id]
            for frame in self._frames
            if speech.start <= frame.time <= speech.end and track_id in frame.looking
        ]
        return sum(looked) / len(looked) if looked else None

    def alone(self, track_id: int, speech: SpeechSpan) -> bool:
        """Whether no face but the one on `track_id` was seen over `speech`."""
        return not any(
            frame.looking.keys() - {track_id}
            for frame in self._frames
            if speech.start <= frame.time <= speech.end
        )


class _Frame(NamedTuple):
    time: float
    looking: dict[int, bool]
    """Whether each face seen in the frame looked at the robot, by track id."""
