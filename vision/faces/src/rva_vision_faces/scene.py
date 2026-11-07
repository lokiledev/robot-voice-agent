from collections.abc import Mapping, Sequence

from rva_pydantic import RVABaseModel
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import Body, BoundingBox, FaceFeatures, Hand, Keypoint
from rva_vision_faces.tracker import Track


class Scene(RVABaseModel):
    """What was seen of the tracked faces in one frame, besides their boxes."""

    faces: Mapping[int, FaceFeatures]
    """Features of each face detected in the frame, by track id."""
    bodies: Mapping[int, Body]
    """The skeleton under each face that has one, by track id."""
    unmatched_bodies: tuple[Body, ...]


def people_in_view(
    tracks: Sequence[Track],
    scene: Scene,
    scale: float,
    width: int,
    height: int,
    caption: tuple[int, str] | None,
) -> PeopleInView:
    """The tracks and what was seen of them, in an image scaled by `scale` to `width` by `height`.

    `caption` is a track id and what that person said, shown on their box.
    """
    return PeopleInView(
        image_width=width,
        image_height=height,
        people=tuple(
            TrackedPerson(
                track_id=track.track_id,
                box=_scale_box(track.box, scale),
                person=None if track.match is None else track.match.person,
                confirmed=track.confirmed,
                said=caption[1] if caption is not None and caption[0] == track.track_id else None,
                face=_scale_face(face, scale)
                if (face := scene.faces.get(track.track_id)) is not None
                else None,
                body=_scale_body(body, scale)
                if (body := scene.bodies.get(track.track_id)) is not None
                else None,
            )
            for track in tracks
        ),
        unmatched_bodies=tuple(_scale_body(body, scale) for body in scene.unmatched_bodies),
    )


def _scale_box(box: BoundingBox, scale: float) -> BoundingBox:
    return BoundingBox(
        x=box.x * scale, y=box.y * scale, width=box.width * scale, height=box.height * scale
    )


def _scale_face(face: FaceFeatures, scale: float) -> FaceFeatures:
    if scale == 1:
        return face
    return face.model_copy(
        update={"landmarks": tuple((x * scale, y * scale) for x, y in face.landmarks)}
    )


def _scale_body(body: Body, scale: float) -> Body:
    if scale == 1:
        return body

    def points(keypoints: tuple[Keypoint | None, ...]) -> tuple[Keypoint | None, ...]:
        return tuple(
            None if k is None else Keypoint(x=k.x * scale, y=k.y * scale, score=k.score)
            for k in keypoints
        )

    return Body(
        box=_scale_box(body.box, scale),
        keypoints=points(body.keypoints),
        left_hand=Hand(keypoints=points(body.left_hand.keypoints)),
        right_hand=Hand(keypoints=points(body.right_hand.keypoints)),
    )
