import math
from collections import deque
from collections.abc import Iterator
from typing import Final, Literal, NamedTuple, assert_never
from uuid import UUID

from rva_attention.config import HabituationConfig, VisualAttentionConfig
from rva_schemas.attention import (
    AttentionCandidate,
    AttentionEntity,
    AttentionFeatures,
    AttentionState,
    AttentionTarget,
    FaceEntity,
    HandEntity,
)
from rva_schemas.observation import VisualObservation
from rva_schemas.record import TrackedPerson
from rva_schemas.vision import Hand, Point3D

_IRISES: Final = (468, 473)
"""The centres of the irises among the face mesh's points."""
_PALM: Final = (0, 1, 5, 9, 13, 17)
"""The wrist and the base of each finger among a hand's points: the palm's outline."""
_PALM_POINTS_NEEDED: Final = 3
_EPSILON_S: Final = 1e-6

type Side = Literal["left", "right"]
type EntityKey = tuple[Literal["face"], int] | tuple[Literal["hand"], int, Side]
"""What a face or hand is remembered as: a hand by its owner's track and its side."""


class VisualAttention:
    """Picks the face or hand in view the robot attends to, frame after frame.

    Fed one observation per frame, in order. Nothing here does I/O: fed a recorded sequence, it
    gives the same states back.
    """

    def __init__(self, config: VisualAttentionConfig) -> None:
        self._config: Final = config
        self._entities: Final[dict[EntityKey, _Entity]] = {}
        self._current: AttentionCandidate | None = None
        """The target as last seen."""
        self._attended_since = 0.0
        self._time: float | None = None

    def observe(self, observation: VisualObservation) -> AttentionState:
        now = observation.time
        elapsed = 0.0 if self._time is None else max(now - self._time, 0.0)
        self._time = now
        self._forget(now)
        sightings = list(_sightings(observation))
        for sighting in sightings:
            self._see(sighting, now)
        target = None if self._current is None else _key(self._current.entity)
        for key, entity in self._entities.items():
            entity.habituate(key == target, elapsed)
        people = {person.track_id: person for person in observation.people.people}
        candidates = sorted(
            (self._score(sighting.key, people, now) for sighting in sightings),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        target = self._select(candidates, now)
        return AttentionState(time=now, target=target, candidates=tuple(candidates))

    def _forget(self, now: float) -> None:
        lifetime = self._config.lifetime_s
        for key in [k for k, e in self._entities.items() if e.last_seen < now - lifetime]:
            del self._entities[key]

    def _see(self, sighting: "_Sighting", now: float) -> None:
        entity = self._entities.get(sighting.key)
        if entity is None:
            entity = self._returned(sighting, now) or _Entity(
                now, self._config.lifetime_s, self._habituation_config(sighting.key)
            )
            self._entities[sighting.key] = entity
        entity.person_id = sighting.person_id
        entity.place(now, sighting.point)

    def _returned(self, sighting: "_Sighting", now: float) -> "_Entity | None":
        """The entity of the same known person on a track no longer seen, moved to this one.

        They are not new: their novelty was spent, and their habituation is theirs.
        """
        if sighting.person_id is None:
            return None
        for key, entity in self._entities.items():
            if (
                key[0] == "face"
                and entity.person_id == sighting.person_id
                and entity.last_seen < now
            ):
                del self._entities[key]
                return entity
        return None

    def _habituation_config(self, key: EntityKey) -> HabituationConfig:
        match key[0]:
            case "face":
                return self._config.face
            case "hand":
                return self._config.hand
            case _:
                assert_never(key[0])

    def _score(
        self, key: EntityKey, people: dict[int, TrackedPerson], now: float
    ) -> AttentionCandidate:
        config = self._config
        entity = self._entities[key]
        point = entity.point
        moving = min(entity.speed(now - config.moving_window_s) / config.speed_m_s, 1.0)
        novelty = math.exp(-(now - entity.first_seen) / config.novelty_decay_s)
        match key:
            case ("face", track_id):
                person = people[track_id]
                assert person.face is not None
                features = AttentionFeatures(
                    closeness=_fade(_norm(point), config.far_m),
                    centrality=_fade(_off_axis(point), config.edge_deg),
                    speaking=float(person.speaker is not None and person.speaker.speaking),
                    eye_contact=_fade(person.face.gaze.off_camera(), config.eye_contact_deg),
                    moving=moving,
                    known=float(person.person is not None),
                    novelty=novelty,
                )
                attended: AttentionEntity = FaceEntity(track_id=track_id, person=person.person)
            case ("hand", track_id, side):
                features = AttentionFeatures(
                    closeness=0.0,
                    centrality=0.0,
                    speaking=0.0,
                    eye_contact=0.0,
                    moving=moving,
                    known=0.0,
                    novelty=novelty,
                )
                attended = HandEntity(track_id=track_id, side=side)
            case _:
                assert_never(key)
        weights = config.weights
        # Over the weights' sum, so the score is from 0 to 1 whatever the weights.
        weighted = sum(
            getattr(weights, name) * getattr(features, name)
            for name in AttentionFeatures.model_fields
        ) / sum(getattr(weights, name) for name in AttentionFeatures.model_fields)
        return AttentionCandidate(
            entity=attended,
            point=point,
            features=features,
            habituation=entity.habituation,
            score=weighted * entity.habituation,
        )

    def _select(self, candidates: list[AttentionCandidate], now: float) -> AttentionTarget | None:
        """The target, kept while in view and good enough, until a better one insists."""
        config = self._config
        current = None if self._current is None else self._seen_again(candidates, now)
        best = candidates[0] if candidates and candidates[0].score >= config.min_score else None
        chosen = current
        if current is None:
            chosen = best
        elif (
            best is not None
            and best is not current
            # Frame times are floats: a dwell a rounding short counts.
            and now - self._attended_since >= config.min_dwell_s - _EPSILON_S
            and best.score > current.score + config.margin
        ):
            chosen = best
        if (None if chosen is None else _key(chosen.entity)) != (
            None if self._current is None else _key(self._current.entity)
        ):
            self._attended_since = now
        self._current = chosen
        return None if chosen is None else AttentionTarget(entity=chosen.entity, point=chosen.point)

    def _seen_again(
        self, candidates: list[AttentionCandidate], now: float
    ) -> AttentionCandidate | None:
        """The current target among this frame's candidates, or held through a short miss.

        A face the mesh or the depth lost for a frame has not left.
        """
        assert self._current is not None
        key = _key(self._current.entity)
        config = self._config
        for candidate in candidates:
            if _key(candidate.entity) == key:
                return candidate if candidate.score >= config.min_score else None
        entity = self._entities.get(key)
        if entity is None or now - entity.last_seen > config.hold_s:
            return None
        return self._current.model_copy(update={"point": entity.point})


class _Sighting(NamedTuple):
    key: EntityKey
    point: Point3D
    person_id: UUID | None


class _Entity:
    """A face or hand remembered, with where it was lately."""

    def __init__(self, first_seen: float, history_s: float, config: HabituationConfig) -> None:
        self.first_seen: Final = first_seen
        self.last_seen = first_seen
        self.habituation = 1.0
        self.person_id: UUID | None = None
        self._config: Final = config
        self._history_s: Final = history_s
        self._poses: Final = deque[tuple[float, Point3D]]()

    @property
    def point(self) -> Point3D:
        return self._poses[-1][1]

    def place(self, now: float, point: Point3D) -> None:
        self.last_seen = now
        self._poses.append((now, point))
        while self._poses[0][0] < now - self._history_s:
            self._poses.popleft()

    def speed(self, since: float) -> float:
        """Metres per second along the path since `since`: a hand waving in place is fast."""
        poses = [(time, point) for time, point in self._poses if time >= since]
        duration = poses[-1][0] - poses[0][0]
        if duration <= 0:
            return 0.0
        path = sum(_distance(a, b) for (_, a), (_, b) in zip(poses, poses[1:], strict=False))
        return path / duration

    def habituate(self, attended: bool, elapsed: float) -> None:
        """Move towards the floor while attended, back towards 1 while not."""
        config = self._config
        if attended:
            rest, towards, tau = self.habituation - config.floor, config.floor, config.decay_s
        else:
            rest, towards, tau = self.habituation - 1.0, 1.0, config.recovery_s
        self.habituation = towards + rest * math.exp(-elapsed / tau)


def _sightings(observation: VisualObservation) -> Iterator[_Sighting]:
    """Every confirmed face and every hand of the main user placed in 3D."""
    for person in observation.people.people:
        if not person.confirmed:
            continue
        face = person.face
        if face is not None and face.in_3d is not None:
            left, right = (face.in_3d.landmarks[iris] for iris in _IRISES)
            person_id = None if person.person is None else person.person.user_id
            yield _Sighting(("face", person.track_id), _mean((left, right)), person_id)
        if person.track_id != observation.main_user or person.body is None:
            continue
        for side, hand in (("left", person.body.left_hand), ("right", person.body.right_hand)):
            palm = None if hand is None else _palm(hand)
            if palm is not None:
                yield _Sighting(("hand", person.track_id, side), palm, None)


def _palm(hand: Hand) -> Point3D | None:
    """The centre of the palm's points placed in 3D, if enough of them are."""
    points = [
        keypoint.position
        for index in _PALM
        if (keypoint := hand.keypoints[index]) is not None and keypoint.position is not None
    ]
    return _mean(points) if len(points) >= _PALM_POINTS_NEEDED else None


def _key(entity: AttentionEntity) -> EntityKey:
    match entity:
        case FaceEntity():
            return ("face", entity.track_id)
        case HandEntity():
            return ("hand", entity.track_id, entity.side)
        case _:
            assert_never(entity)


def _fade(value: float, limit: float) -> float:
    """1 at 0, falling to 0 at `limit` and beyond."""
    return max(1.0 - value / limit, 0.0)


def _norm(point: Point3D) -> float:
    return math.sqrt(point.x**2 + point.y**2 + point.z**2)


def _off_axis(point: Point3D) -> float:
    """Angle between the camera's axis and the point, in degrees."""
    norm = _norm(point)
    return 90.0 if norm == 0 else math.degrees(math.acos(max(min(point.z / norm, 1.0), -1.0)))


def _distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _mean(points: tuple[Point3D, ...] | list[Point3D]) -> Point3D:
    n = len(points)
    return Point3D(
        x=sum(p.x for p in points) / n,
        y=sum(p.y for p in points) / n,
        z=sum(p.z for p in points) / n,
    )
