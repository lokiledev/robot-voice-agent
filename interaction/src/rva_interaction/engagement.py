import math
from collections import deque
from typing import Final

from rva_interaction.config import EngagementConfig
from rva_schemas.interaction import (
    Disengaged,
    Engaged,
    EngagementEvent,
    EngagementState,
    PersonEngagement,
)
from rva_schemas.observation import VisualObservation
from rva_schemas.record import TrackedPerson
from rva_schemas.vision import KnownPerson, Point3D

_IRISES: Final = (468, 473)
"""The centres of the irises among the face mesh's points."""
_MAX_STEP_S: Final = 0.2
"""The most a frame counts for: what a person did while no frame came, as the models load or the
camera stalls, is not known."""


class EngagementTracker:
    """Tells who in view pays attention to the robot, frame after frame.

    Fed one observation per frame, in order. It runs on the frames' times and does no I/O: fed a
    recorded sequence, it gives the same states back.
    """

    def __init__(self, config: EngagementConfig, eye_contact_deg: float) -> None:
        self._config: Final = config
        self._eye_contact_deg: Final = eye_contact_deg
        self._people: Final[dict[int, _Person]] = {}
        self._pending: Final[list[EngagementEvent]] = []
        """Events of people engaged between frames, told with the next one."""

    @property
    def engaged(self) -> frozenset[int]:
        return frozenset(track_id for track_id, person in self._people.items() if person.engaged)

    def engage(self, track_id: int) -> None:
        """Take the person on `track_id` as engaged from now on, e.g. as they spoke to the robot.

        Nothing changes for a track no longer in view, or with `max_engaged` people engaged.
        """
        person = self._people.get(track_id)
        if person is None or person.engaged or len(self.engaged) >= self._config.max_engaged:
            return
        person.score = 1.0
        person.engaged = True
        self._pending.append(Engaged(track_id=track_id, person=person.known))

    def observe(self, observation: VisualObservation) -> EngagementState:
        now = observation.time
        events, self._pending[:] = list(self._pending), []
        seen = {person.track_id: person for person in observation.people.people}
        for track_id in self._people.keys() - seen.keys():
            gone = self._people.pop(track_id)
            if gone.engaged:
                events.append(Disengaged(track_id=track_id, person=gone.known, left=True))
        states = [self._follow(person, now, events) for person in seen.values()]
        return EngagementState(time=now, people=tuple(states), events=tuple(events))

    def _follow(
        self, seen: TrackedPerson, now: float, events: list[EngagementEvent]
    ) -> PersonEngagement:
        config = self._config
        person = self._people.setdefault(seen.track_id, _Person(config.moving_window_s, now))
        person.known = seen.person
        # Since this person was last seen, and nothing at first sight.
        elapsed = min(max(now - person.last_seen, 0.0), _MAX_STEP_S)
        person.last_seen = now
        point = _between_the_eyes(seen)
        if point is None:
            # Without depth, any face large enough to be tracked is taken as close and staying.
            closeness = staying = 1.0
        else:
            person.place(now, point)
            closeness = _fade(_norm(point) - config.near_m, config.far_m - config.near_m)
            staying = _fade(person.speed(), config.walking_m_s)
        looking = seen.face is not None and seen.face.gaze.off_camera() <= self._eye_contact_deg
        # Someone engaged who speaks looks away to think, and has not turned away for that.
        speaking = person.engaged and seen.speaker is not None and seen.speaker.speaking
        attending = looking or speaking
        interest = closeness * staying if attending else 0.0
        tau = config.rise_s if interest > person.score else config.fall_s
        person.score = interest + (person.score - interest) * math.exp(-elapsed / tau)
        # The score builds from a face's first frame, but only a confirmed face engages.
        if (
            not person.engaged
            and seen.confirmed
            and person.score >= config.enter
            and len(self.engaged) < config.max_engaged
        ):
            person.engaged = True
            events.append(Engaged(track_id=seen.track_id, person=seen.person))
        elif person.engaged and person.score <= config.leave:
            person.engaged = False
            events.append(Disengaged(track_id=seen.track_id, person=seen.person, left=False))
        return PersonEngagement(
            track_id=seen.track_id,
            closeness=closeness,
            staying=staying,
            attending=attending,
            score=person.score,
            engaged=person.engaged,
        )


class _Person:
    def __init__(self, window_s: float, first_seen: float) -> None:
        self.last_seen = first_seen
        self.score = 0.0
        self.engaged = False
        self.known: KnownPerson | None = None
        self._window_s: Final = window_s
        self._poses: Final = deque[tuple[float, Point3D]]()

    def place(self, now: float, point: Point3D) -> None:
        self._poses.append((now, point))
        while self._poses[0][0] < now - self._window_s:
            self._poses.popleft()

    def speed(self) -> float:
        """Metres per second from where they were a window ago to where they are.

        Straight between the two, not along the path: the depth of a face still jitters.
        """
        (then, start), (now, end) = self._poses[0], self._poses[-1]
        return 0.0 if now <= then else _norm(_minus(end, start)) / (now - then)


def _between_the_eyes(person: TrackedPerson) -> Point3D | None:
    if person.face is None or person.face.in_3d is None:
        return None
    left, right = (person.face.in_3d.landmarks[iris] for iris in _IRISES)
    return Point3D(x=(left.x + right.x) / 2, y=(left.y + right.y) / 2, z=(left.z + right.z) / 2)


def _fade(value: float, limit: float) -> float:
    """1 at 0 and under, falling to 0 at `limit` and beyond."""
    return min(max(1.0 - value / limit, 0.0), 1.0)


def _norm(point: Point3D) -> float:
    return math.sqrt(point.x**2 + point.y**2 + point.z**2)


def _minus(a: Point3D, b: Point3D) -> Point3D:
    return Point3D(x=a.x - b.x, y=a.y - b.y, z=a.z - b.z)
