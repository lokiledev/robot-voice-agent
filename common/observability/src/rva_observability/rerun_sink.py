import math
from datetime import UTC, datetime
from typing import Final, assert_never, override

import rerun as rr
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rva_body_schemas.motion import describe
from rva_schemas.record import (
    AgentStarted,
    CameraFrame,
    FaceUpdate,
    GazeUpdate,
    Measurement,
    MotionUpdate,
    MuteUpdate,
    PeopleInView,
    Record,
    RobotLine,
    RobotSentence,
    Speaker,
    TaskUpdate,
    ToolInvocation,
    TrackedPerson,
    UserLine,
)
from rva_schemas.vision import BODY_CONNECTIONS, HAND_CONNECTIONS, Body, FaceFeatures, Keypoint

_TIMELINE: Final = "wall"
_CAMERA: Final = "camera"
"""Image and people share this entity, so rerun draws them in one 2D view."""
_PEOPLE: Final = f"{_CAMERA}/people"
"""One entity per face track, under which its face, skeleton and hands hang."""
_BODY_CLASS: Final = 1
_HAND_CLASS: Final = 2
# RGB, as in the debug window.
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""


class RerunRecorder:
    """`Recorder` writing each entry to rerun, on the wall-clock timeline."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording
        self._recording.log(
            _CAMERA,
            rr.AnnotationContext(
                [
                    rr.ClassDescription(
                        info=(_BODY_CLASS, "body"), keypoint_connections=BODY_CONNECTIONS
                    ),
                    rr.ClassDescription(
                        info=(_HAND_CLASS, "hand"), keypoint_connections=HAND_CONNECTIONS
                    ),
                ]
            ),
            static=True,
        )
        # Entities of the last frame's people, cleared once they are not seen.
        self._people_entities: set[str] = set()

    def record(self, entry: Record) -> None:
        self._recording.set_time(_TIMELINE, timestamp=datetime.now(UTC))
        match entry:
            case AgentStarted():
                tools = ", ".join(tool.name for tool in entry.tools)
                self._recording.log("agent/started", rr.TextLog(f"{entry.name}, tools: {tools}"))
            case UserLine():
                text = f"[{_who(entry.speaker)}] {entry.text}"
                self._recording.log("dialogue/user", rr.TextLog(text))
            case RobotSentence():
                self._recording.log("dialogue/robot/sentences", rr.TextLog(entry.text))
            case RobotLine():
                text = f"{entry.text} [interrupted]" if entry.interrupted else entry.text
                self._recording.log("dialogue/robot", rr.TextLog(text))
            case ToolInvocation():
                text = f"{entry.name}({entry.arguments}) -> {entry.result}"
                self._recording.log("agent/tools", rr.TextLog(text))
            case TaskUpdate():
                level = "ERROR" if entry.task.status.state == "failed" else "INFO"
                text = f"{entry.task.instruction!r}: {entry.task.status.model_dump_json()}"
                self._recording.log("vla/task", rr.TextLog(text, level=level))
            case MotionUpdate():
                level = "ERROR" if entry.motion.status.state == "failed" else "INFO"
                text = f"{describe(entry.motion.action)}: {entry.motion.status.model_dump_json()}"
                self._recording.log("body/motion", rr.TextLog(text, level=level))
            case GazeUpdate():
                person = entry.looking_at
                text = (
                    "stopped looking"
                    if person is None
                    else f"looking at {person.name} ({person.user_id})"
                )
                self._recording.log("body/gaze", rr.TextLog(text))
            case MuteUpdate():
                self._recording.log("agent/muted", rr.TextLog("muted" if entry.muted else "talks"))
            case Measurement():
                self._recording.log(f"measurements/{entry.name}", rr.Scalars(entry.value))
            case FaceUpdate():
                self._recording.log("vision/faces", rr.TextLog(entry.event.model_dump_json()))
            case PeopleInView():
                self._people(entry)
            case CameraFrame():
                self._recording.log(
                    f"{_CAMERA}/image",
                    rr.EncodedImage(contents=entry.jpeg, media_type="image/jpeg"),
                )
            case _:
                assert_never(entry)

    def _people(self, entry: PeopleInView) -> None:
        entities: dict[str, rr.AsComponents] = {}
        for person in entry.people:
            path = f"{_PEOPLE}/{person.track_id}"
            entities[f"{path}/box"] = rr.Boxes2D(
                array=[[person.box.x, person.box.y, person.box.width, person.box.height]],
                array_format=rr.Box2DFormat.XYWH,
                labels=[_label(person)],
                colors=[_color(person)],
                show_labels=True,
            )
            if person.face is not None:
                entities.update(_face(path, person.face, _color(person)))
                self._face_series(person.track_id, person.face)
            if person.body is not None:
                entities.update(_body(f"{path}/body", person.body, _color(person)))
        for index, body in enumerate(entry.unmatched_bodies):
            entities.update(_body(f"{_CAMERA}/unmatched_bodies/{index}", body, _UNCONFIRMED))
        for path, archetype in entities.items():
            self._recording.log(path, archetype)
        for path in self._people_entities - entities.keys():
            self._recording.log(path, rr.Clear(recursive=False))
        self._people_entities = set(entities)

    def _face_series(self, track_id: int, face: FaceFeatures) -> None:
        """Plot what changes fast, e.g. the mouth moving while someone speaks."""
        path = f"people/{track_id}"
        for name, value in (
            ("eyes/left", face.left_eye_open),
            ("eyes/right", face.right_eye_open),
            ("mouth", face.mouth_open),
            ("gaze/yaw", face.gaze.yaw),
            ("gaze/pitch", face.gaze.pitch),
        ):
            self._recording.log(f"{path}/{name}", rr.Scalars(value))


def _face(path: str, face: FaceFeatures, color: tuple[int, int, int]) -> dict[str, rr.AsComponents]:
    (left_x, left_y), (right_x, right_y) = (face.landmarks[iris] for iris in _IRISES)
    between_eyes = ((left_x + right_x) / 2, (left_y + right_y) / 2)
    yaw, pitch = math.radians(face.gaze.yaw), math.radians(face.gaze.pitch)
    xs = [x for x, _ in face.landmarks]
    length = max(xs) - min(xs)
    return {
        f"{path}/face": rr.Points2D(face.landmarks, colors=color, radii=0.5),
        f"{path}/gaze": rr.Arrows2D(
            origins=[between_eyes],
            vectors=[[length * math.sin(yaw), -length * math.sin(pitch)]],
            colors=[(0, 255, 255)],
        ),
    }


def _body(path: str, body: Body, color: tuple[int, int, int]) -> dict[str, rr.AsComponents]:
    return {
        path: _skeleton(body.keypoints, _BODY_CLASS, color),
        f"{path}/left_hand": _skeleton(body.left_hand.keypoints, _HAND_CLASS, color),
        f"{path}/right_hand": _skeleton(body.right_hand.keypoints, _HAND_CLASS, color),
    }


def _skeleton(
    keypoints: tuple[Keypoint | None, ...], class_id: int, color: tuple[int, int, int]
) -> rr.Points2D:
    visible = [(index, point) for index, point in enumerate(keypoints) if point is not None]
    return rr.Points2D(
        [(point.x, point.y) for _, point in visible],
        keypoint_ids=[index for index, _ in visible],
        class_ids=class_id,
        colors=color,
        radii=2.0,
    )


def _who(speaker: Speaker | None) -> str:
    if speaker is None:
        return "nobody in view"
    if speaker.person is None:
        return f"unknown, track {speaker.track_id}"
    role = ", operator" if speaker.operator else ""
    return f"{speaker.person.name}, track {speaker.track_id}{role}"


def _label(person: TrackedPerson) -> str:
    who = (
        "unknown"
        if person.person is None
        else f"{person.person.name} ({str(person.person.user_id)[:8]})"
    )
    face = "" if person.face is None else f", {person.face.expression}"
    said = "" if person.said is None else f': "{person.said}"'
    return f"#{person.track_id} {who}{face}{said}"


def _color(person: TrackedPerson) -> tuple[int, int, int]:
    if not person.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if person.person is None else _KNOWN


class RerunSpanProcessor(SpanProcessor):
    """Logs the duration of each finished span to `latency/<span name>`, in milliseconds.

    A span event logs the time from the start of its span to `latency/<span name>/<event name>`,
    e.g. the time to the first token.
    """

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording

    @override
    def on_end(self, span: ReadableSpan) -> None:
        if span.start_time is None or span.end_time is None:
            return
        self._recording.set_time(_TIMELINE, timestamp=span.end_time * 1e-9)
        name = span.name.replace(":", "/")
        self._recording.log(f"latency/{name}", rr.Scalars((span.end_time - span.start_time) * 1e-6))
        for event in span.events:
            elapsed_ms = (event.timestamp - span.start_time) * 1e-6
            self._recording.log(f"latency/{name}/{event.name}", rr.Scalars(elapsed_ms))
