import math
import textwrap
from collections.abc import Mapping
from itertools import pairwise
from typing import Final, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from rva_schemas.attention import AttentionTarget
from rva_schemas.interaction import PersonEngagement
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import (
    BODY_CONNECTIONS,
    FINGERS,
    Body,
    FaceFeatures,
    Keypoint,
    VisionTimings,
)

type Image = npt.NDArray[np.uint8]
"""BGR, height x width x 3."""

# BGR
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (0, 165, 255)
_UNCONFIRMED: Final = (160, 160, 160)
_ATTENTION: Final = (255, 0, 255)
"""The point attended to, as in rerun."""
_WHITE: Final = (255, 255, 255)
_SPEECH_LINE_CHARS: Final = 40
# Face mesh outlines, from MediaPipe's face_mesh_connections: both eyes, then the lips.
_FACE_CONTOURS: Final = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191),
)  # fmt: skip
# BGR colors of the thumb, index, middle, ring and little finger: cool for the person's left hand
# (blue, sky blue, aqua, violet, lavender), warm for their right (red, orange, yellow, pink,
# magenta).
_LEFT_FINGER_COLORS: Final = (
    (255, 60, 30),
    (255, 170, 0),
    (230, 255, 0),
    (255, 80, 150),
    (255, 190, 210),
)
_RIGHT_FINGER_COLORS: Final = (
    (40, 40, 255),
    (0, 140, 255),
    (0, 230, 255),
    (180, 105, 255),
    (220, 0, 220),
)
_IRISES: Final = (468, 473)
"""Centers of the irises in the face mesh."""


class OverlayOptions(NamedTuple):
    """What is drawn over the camera image besides each person's box, name and words."""

    bodies: bool
    """The skeletons, without their hands."""
    hands: bool
    faces: bool
    """The outline of the eyes and lips, the irises and the gaze."""
    engagement: bool
    """Each person's engagement score, and a doubled box around an engaged one."""


EVERYTHING: Final = OverlayOptions(bodies=True, hands=True, faces=True, engagement=True)


def draw(
    frame: Image,
    people: PeopleInView,
    timings: VisionTimings,
    attention: AttentionTarget | None,
    engagement: Mapping[int, PersonEngagement],
    options: OverlayOptions,
) -> None:
    """Draw the people and what was seen of them, the timings, and the user's last sentence.

    `people` is in the frame's pixels. Each person gets their box and label, what they said
    under their box, and what `options` asks for of their face, skeleton and hands. Skeletons
    without a face are drawn in gray. `attention` is the point attended to, marked where the camera's intrinsics put it in
    the frame, if they are known. `engagement` is how much each person pays attention to the
    robot, by track id: its score is on their label, and an engaged person's box is doubled.
    """
    scale = frame.shape[0] / 540
    thickness = max(1, round(2 * scale))
    for body in people.unmatched_bodies:
        _draw_body(frame, body, _UNCONFIRMED, thickness, options)
    for person in people.people:
        box = person.box
        x, y = round(box.x), round(box.y)
        right, bottom = round(box.x + box.width), round(box.y + box.height)
        color = color_of(person)
        if person.body is not None:
            _draw_body(frame, person.body, color, thickness, options)
        cv2.rectangle(frame, (x, y), (right, bottom), color, thickness)
        lines = [label(person)]
        engaging = engagement.get(person.track_id) if options.engagement else None
        if engaging is not None:
            lines.append(describe_engagement(engaging))
            if engaging.engaged:
                gap = 3 * thickness
                # Inside the box, clear of the label above it.
                cv2.rectangle(
                    frame, (x + gap, y + gap), (right - gap, bottom - gap), color, thickness
                )
        if person.face is not None and options.faces:
            _draw_face(frame, person.face, box.width, color, thickness)
        for index, line in enumerate(reversed(lines)):
            cv2.putText(
                frame,
                line,
                (x, max(y - round((8 + 22 * index) * scale), round(20 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale,
                color,
                thickness,
            )
        if person.said is not None:
            _draw_speech(frame, person.said, (x, bottom), color, scale, thickness)
    overlay = (
        f"{timings.fps:.0f} fps  capture {timings.capture_ms:.1f} ms  "
        f"detect {timings.detect_ms:.1f} ms  embed {timings.embed_ms:.1f} ms  "
        f"faces {timings.faces_ms:.1f} ms  bodies {timings.bodies_ms:.1f} ms  "
        f"depth {timings.depth_ms:.1f} ms  speaker {timings.speaker_ms:.1f} ms"
    )
    cv2.putText(
        frame,
        overlay,
        (round(10 * scale), round(25 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        (255, 255, 255),
        thickness,
    )
    if attention is not None and people.intrinsics is not None:
        _draw_attention(frame, attention, people, scale, thickness)


def _draw_attention(
    frame: Image,
    attention: AttentionTarget,
    people: PeopleInView,
    scale: float,
    thickness: int,
) -> None:
    """A pair of eyes looking down at the point attended to.

    Above a face's labels, or a little above a hand.
    """
    assert people.intrinsics is not None
    pixel = people.intrinsics.project(attention.point)
    if pixel is None:
        return
    x, y = pixel
    above = y - 70 * scale
    if attention.entity.kind == "face":
        track_id = attention.entity.track_id
        box = next((p.box for p in people.people if p.track_id == track_id), None)
        if box is not None:
            above = box.y - 60 * scale
    radius_x, radius_y = 16 * scale, 11 * scale
    pupil = max(1, round(4 * scale))
    for side in (-1, 1):
        centre = (x + side * 20 * scale, above)
        cv2.ellipse(
            frame, _point(*centre), (round(radius_x), round(radius_y)), 0, 0, 360, _WHITE, -1
        )
        cv2.ellipse(
            frame,
            _point(*centre),
            (round(radius_x), round(radius_y)),
            0,
            0,
            360,
            _ATTENTION,
            thickness,
        )
        # The pupils look at the point from where the eyes are.
        dx, dy = x - centre[0], y - centre[1]
        norm = max(math.hypot(dx, dy), 1.0)
        reach_x, reach_y = radius_x - pupil - thickness, radius_y - pupil - thickness
        gaze = (centre[0] + dx / norm * reach_x, centre[1] + dy / norm * reach_y)
        cv2.circle(frame, _point(*gaze), pupil, _ATTENTION, -1)


def describe_engagement(engagement: PersonEngagement) -> str:
    """Whether a person is engaged, with their score."""
    return f"{'engaged' if engagement.engaged else 'not engaged'} {engagement.score:.2f}"


def label(person: TrackedPerson) -> str:
    if person.person is None:
        return f"#{person.track_id} unknown"
    who = person.person
    return f"#{person.track_id} {who.name} ({str(who.user_id)[:8]}) {person.match_score:.2f}"


def color_of(person: TrackedPerson) -> tuple[int, int, int]:
    """BGR color of a track: gray until announced, then green if recognized, else orange."""
    if not person.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if person.person is None else _KNOWN


def _draw_speech(
    frame: Image,
    text: str,
    under: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    """Draw a sentence in a dark box hanging under a face box, wrapped to a few lines."""
    font_scale = 0.6 * scale
    lines = textwrap.wrap(f'"{text}"', _SPEECH_LINE_CHARS)
    (_, line_height), baseline = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        for line in lines
    ]
    pad, step = round(6 * scale), line_height + baseline + round(4 * scale)
    height, width = frame.shape[:2]
    # Keep the box inside the image, under the face if there is room, else above its bottom edge.
    box_width, box_height = max(widths) + 2 * pad, step * len(lines) + pad
    left = min(max(under[0], 0), max(width - box_width, 0))
    top = min(under[1] + round(4 * scale), max(height - box_height, 0))
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), color, thickness)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (left + pad, top + pad + line_height + index * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def _draw_body(
    frame: Image,
    body: Body,
    color: tuple[int, int, int],
    thickness: int,
    options: OverlayOptions,
) -> None:
    """Draw the body in the track's color, and each finger of the hands in a color of its own."""
    if options.bodies:
        _draw_bones(frame, body.keypoints, BODY_CONNECTIONS, color, thickness)
    if not options.hands:
        return
    for hand, colors in (
        (body.left_hand, _LEFT_FINGER_COLORS),
        (body.right_hand, _RIGHT_FINGER_COLORS),
    ):
        if hand is None:
            continue
        for finger, finger_color in zip(FINGERS, colors, strict=True):
            _draw_bones(frame, hand.keypoints, tuple(pairwise(finger)), finger_color, thickness)


def _draw_bones(
    frame: Image,
    keypoints: tuple[Keypoint | None, ...],
    bones: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in bones:
        a, b = keypoints[start], keypoints[end]
        if a is not None and b is not None:
            cv2.line(frame, _point(a.x, a.y), _point(b.x, b.y), color, thickness)
    for index in {index for bone in bones for index in bone}:
        if (keypoint := keypoints[index]) is not None:
            cv2.circle(frame, _point(keypoint.x, keypoint.y), thickness + 1, color, -1)


def _draw_face(
    frame: Image, face: FaceFeatures, size: float, color: tuple[int, int, int], thickness: int
) -> None:
    """Outline the eyes and lips, mark the irises, and draw the gaze from between the eyes."""
    points = np.array(face.landmarks, np.float32)
    for contour in _FACE_CONTOURS:
        cv2.polylines(frame, [points[list(contour)].round().astype(np.int32)], True, color, 1)
    for iris in _IRISES:
        cv2.circle(frame, _point(*points[iris]), thickness, (255, 255, 255), -1)
    origin = points[list(_IRISES)].mean(axis=0)
    yaw, pitch = np.radians(face.gaze.yaw), np.radians(face.gaze.pitch)
    tip = origin + size * np.array([np.sin(yaw), -np.sin(pitch)])
    cv2.arrowedLine(frame, _point(*origin), _point(*tip), (255, 255, 0), thickness, tipLength=0.2)


def _point(x: float, y: float) -> tuple[int, int]:
    return round(x), round(y)
