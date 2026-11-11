from rva_schemas.observation import VisualObservation
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import (
    ActiveSpeaker,
    BoundingBox,
    Face3D,
    FaceFeatures,
    Gaze,
    GazeRay,
    Point3D,
)

AT_ROBOT = Gaze(yaw=0.0, pitch=0.0)
AWAY = Gaze(yaw=40.0, pitch=0.0)


def at(x: float, z: float) -> Point3D:
    return Point3D(x=x, y=0.0, z=z)


def face(gaze: Gaze, point: Point3D | None) -> FaceFeatures:
    """A face looking along `gaze`, whose every 3D landmark is at `point`; not in 3D without one."""
    return FaceFeatures(
        landmarks=((0.0, 0.0),) * 478,
        left_eye_open=1.0,
        right_eye_open=1.0,
        mouth_open=0.0,
        gaze=gaze,
        in_3d=None
        if point is None
        else Face3D(
            landmarks=(point,) * 478,
            gaze=GazeRay(origin=point, direction=Point3D(x=0.0, y=0.0, z=-1.0)),
        ),
        expression="neutral",
        expression_score=1.0,
    )


def seen(track_id: int, features: FaceFeatures | None, *, speaking: bool) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
        person=None,
        match_score=None,
        confirmed=True,
        said=None,
        face=features,
        body=None,
        speaker=ActiveSpeaker(score=1.0 if speaking else -3.0, speaking=speaking),
    )


def frame(time: float, *people: TrackedPerson) -> VisualObservation:
    return VisualObservation(
        time=time,
        people=PeopleInView(
            frame="camera_optical",
            image_width=1920,
            image_height=1080,
            intrinsics=None,
            people=people,
            unmatched_bodies=(),
        ),
        main_user=None,
    )
