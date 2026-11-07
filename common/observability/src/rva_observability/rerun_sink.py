import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, assert_never, override

import numpy as np
import rerun as rr
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rva_body_schemas.motion import describe
from rva_schemas.record import (
    AgentStarted,
    CameraFrame,
    DepthFrame,
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
from rva_schemas.vision import (
    BODY_CONNECTIONS,
    FINGERS,
    Body,
    CameraIntrinsics,
    FaceFeatures,
    GazeRay,
    Hand,
    Keypoint,
    Point3D,
)

_TIMELINE: Final = "wall"
_CAMERA: Final = "camera"
"""Image and people share this entity, so rerun draws them in one 2D view; with the camera's
intrinsics, it is also the camera's pinhole in the 3D view."""
_PEOPLE: Final = f"{_CAMERA}/people"
"""One entity per face track, under which its face, skeleton and hands hang."""
_SPACE: Final = "space"
"""The people in 3D, in the camera's frame, which is the 3D view's."""
_PEOPLE_3D: Final = f"{_SPACE}/people"
_DEPTH: Final = "depth"
"""The depth's pinhole, at its own resolution, with the depth under it as a point cloud."""
_DEPTH_STRIDE: Final = 2
_GAZE_LENGTH_M: Final = 0.5
_BODY_CLASS: Final = 1
# RGB, as in the debug window.
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)
# Thumb, index, middle, ring and little finger, as in the debug window: cool for the person's
# left hand, warm for their right.
_LEFT_FINGER_COLORS: Final = (
    (30, 60, 255),
    (0, 170, 255),
    (0, 255, 230),
    (150, 80, 255),
    (210, 190, 255),
)
_RIGHT_FINGER_COLORS: Final = (
    (255, 40, 40),
    (255, 140, 0),
    (255, 230, 0),
    (255, 105, 180),
    (220, 0, 220),
)
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""


class RerunRecorder:
    """`Recorder` writing each entry to rerun, on the wall-clock timeline."""

    def __init__(self, recording: rr.RecordingStream) -> None:
        self._recording: Final = recording
        self._recording.log("/", rr.ViewCoordinates.RDF, static=True)
        for path in (_CAMERA, _SPACE):
            self._recording.log(
                path,
                rr.AnnotationContext(
                    [
                        rr.ClassDescription(
                            info=(_BODY_CLASS, "body"), keypoint_connections=BODY_CONNECTIONS
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
            case DepthFrame():
                # Every other pixel, in millimetres on 16 bits: an eighth of the size, about
                # 2 MB/s, and still enough points to see the people against the room.
                metres = entry.depth.metres[::_DEPTH_STRIDE, ::_DEPTH_STRIDE]
                rows, columns = metres.shape
                intrinsics = entry.intrinsics
                strided = intrinsics.model_copy(
                    update={
                        "width": columns,
                        "height": rows,
                        **{
                            name: getattr(intrinsics, name) / _DEPTH_STRIDE
                            for name in ("fx", "fy", "cx", "cy")
                        },
                    }
                )
                self._recording.log(_DEPTH, _pinhole(strided))
                millimetres = np.clip(metres * 1000, 0, np.iinfo(np.uint16).max)
                self._recording.log(
                    f"{_DEPTH}/image", rr.DepthImage(millimetres.astype(np.uint16), meter=1000)
                )
            case _:
                assert_never(entry)

    def _people(self, entry: PeopleInView) -> None:
        if entry.intrinsics is not None:
            self._recording.log(_CAMERA, _pinhole(entry.intrinsics))
        entities: dict[str, rr.AsComponents] = {}
        for person in entry.people:
            path = f"{_PEOPLE}/{person.track_id}"
            path_3d = f"{_PEOPLE_3D}/{person.track_id}"
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
                if person.face.gaze_ray is not None:
                    entities[f"{path_3d}/gaze"] = _gaze_3d(person.face.gaze_ray, _label(person))
            if person.body is not None:
                entities.update(_body(f"{path}/body", person.body, _color(person)))
                entities.update(_body_3d(f"{path_3d}/body", person.body, _color(person)))
            if person.speaker is not None:
                self._recording.log(
                    f"people/{person.track_id}/speaking", rr.Scalars(person.speaker.score)
                )
        for index, body in enumerate(entry.unmatched_bodies):
            entities.update(_body(f"{_CAMERA}/unmatched_bodies/{index}", body, _UNCONFIRMED))
            entities.update(_body_3d(f"{_SPACE}/unmatched_bodies/{index}", body, _UNCONFIRMED))
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
    entities: dict[str, rr.AsComponents] = {path: _skeleton(body.keypoints, _BODY_CLASS, color)}
    for name, hand, colors in _hands(body):
        strips, strip_colors = _fingers(hand, colors, _pixel)
        entities[f"{path}/{name}"] = rr.LineStrips2D(strips, colors=strip_colors, radii=1.0)
    return entities


def _body_3d(path: str, body: Body, color: tuple[int, int, int]) -> dict[str, rr.AsComponents]:
    """The skeleton and hands in the camera's frame, where their points have a position."""
    placed = [
        (index, point.position)
        for index, point in enumerate(body.keypoints)
        if point is not None and point.position is not None
    ]
    entities: dict[str, rr.AsComponents] = {}
    if placed:
        entities[path] = rr.Points3D(
            [_xyz(position) for _, position in placed],
            keypoint_ids=[index for index, _ in placed],
            class_ids=_BODY_CLASS,
            colors=color,
            radii=0.015,
        )
    for name, hand, colors in _hands(body):
        strips, strip_colors = _fingers(hand, colors, _position)
        if strips:
            entities[f"{path}/{name}"] = rr.LineStrips3D(strips, colors=strip_colors, radii=0.004)
    return entities


def _gaze_3d(ray: GazeRay, label: str) -> rr.Arrows3D:
    direction = _xyz(ray.direction)
    return rr.Arrows3D(
        origins=[_xyz(ray.origin)],
        vectors=[tuple(_GAZE_LENGTH_M * axis for axis in direction)],
        colors=[(0, 255, 255)],
        labels=[label],
        radii=0.005,
    )


def _pinhole(intrinsics: CameraIntrinsics) -> rr.Pinhole:
    return rr.Pinhole(
        focal_length=[intrinsics.fx, intrinsics.fy],
        principal_point=[intrinsics.cx, intrinsics.cy],
        width=intrinsics.width,
        height=intrinsics.height,
        camera_xyz=rr.ViewCoordinates.RDF,
    )


def _hands(body: Body) -> list[tuple[str, Hand, tuple[tuple[int, int, int], ...]]]:
    return [
        (name, hand, colors)
        for name, hand, colors in (
            ("left_hand", body.left_hand, _LEFT_FINGER_COLORS),
            ("right_hand", body.right_hand, _RIGHT_FINGER_COLORS),
        )
        if hand is not None
    ]


def _fingers[T](
    hand: Hand,
    finger_colors: tuple[tuple[int, int, int], ...],
    coordinates: Callable[[Keypoint], T | None],
) -> tuple[list[list[T]], list[tuple[int, int, int]]]:
    """Each finger as a strip from the wrist to its tip, in its own color.

    A strip breaks where a point was not found, or `coordinates` has none for it.
    """
    strips: list[list[T]] = []
    colors: list[tuple[int, int, int]] = []
    for finger, color in zip(FINGERS, finger_colors, strict=True):
        strip: list[T] = []
        for point in (*(hand.keypoints[index] for index in finger), None):
            at = None if point is None else coordinates(point)
            if at is not None:
                strip.append(at)
                continue
            if len(strip) > 1:
                strips.append(strip)
                colors.append(color)
            strip = []
    return strips, colors


def _pixel(point: Keypoint) -> tuple[float, float]:
    return point.x, point.y


def _position(point: Keypoint) -> tuple[float, float, float] | None:
    return None if point.position is None else _xyz(point.position)


def _xyz(point: Point3D) -> tuple[float, float, float]:
    return point.x, point.y, point.z


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
