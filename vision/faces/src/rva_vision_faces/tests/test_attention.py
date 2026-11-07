import pytest

from rva_schemas.vision import BoundingBox, Gaze
from rva_vision_faces.attention import Attention, off_camera, prominence
from rva_vision_faces.config import EyeContactConfig

CONFIG = EyeContactConfig(window_s=0.4, enter_deg=12.0, leave_deg=18.0)
CENTER = BoundingBox(x=900, y=300, width=120, height=150)
"""A face in the middle of a 1920 x 1080 frame."""
LOOKING = Gaze(yaw=3.0, pitch=-2.0)
AWAY = Gaze(yaw=40.0, pitch=0.0)


def test_off_camera_is_the_angle_from_looking_straight_at_it() -> None:
    assert off_camera(Gaze(yaw=0.0, pitch=0.0)) == 0.0
    assert off_camera(Gaze(yaw=-15.0, pitch=0.0)) == pytest.approx(15.0)
    assert off_camera(Gaze(yaw=0.0, pitch=20.0)) == pytest.approx(20.0)


def test_prominence_favors_a_central_face_over_a_larger_one_at_the_edge() -> None:
    edge = BoundingBox(x=1650, y=300, width=160, height=200)

    assert prominence(CENTER, 1920, 1080) == pytest.approx(150 / 1080)
    assert prominence(edge, 1920, 1080) == pytest.approx(200 / 1080 * (1 - 770 / 960))
    assert prominence(BoundingBox(x=1900, y=0, width=200, height=200), 1920, 1080) == 0.0


def test_eye_contact_holds_through_a_blink_and_ends_when_the_gaze_turns_away() -> None:
    attention = Attention(CONFIG)
    gazes = [LOOKING, LOOKING, LOOKING, AWAY, LOOKING, AWAY, AWAY, AWAY, AWAY]

    contact = [
        attention.update(0.1 * frame, {7: (CENTER, gaze)}, 1920, 1080)
        for frame, gaze in enumerate(gazes)
    ]

    assert [user.eye_contact if user else None for user in contact] == [
        True, True, True, True, True, True, False, False, False,
    ]  # fmt: skip


def test_eye_contact_starts_within_the_tighter_threshold_and_ends_beyond_the_looser() -> None:
    attention = Attention(CONFIG.model_copy(update={"window_s": 0.0}))

    def look(time: float, yaw: float) -> bool:
        user = attention.update(time, {7: (CENTER, Gaze(yaw=yaw, pitch=0.0))}, 1920, 1080)
        assert user is not None
        return user.eye_contact

    assert [
        look(0.1 * frame, yaw) for frame, yaw in enumerate((15.0, 10.0, 15.0, 17.0, 19.0, 15.0))
    ] == [
        False, True, True, True, False, False,
    ]  # fmt: skip


def test_the_main_user_stays_until_someone_clearly_more_prominent_comes() -> None:
    attention = Attention(CONFIG)
    beside = BoundingBox(x=700, y=300, width=120, height=160)
    closer = BoundingBox(x=700, y=300, width=200, height=260)

    first = attention.update(0.0, {7: (CENTER, LOOKING)}, 1920, 1080)
    slightly_better = attention.update(0.1, {7: (CENTER, LOOKING), 8: (beside, AWAY)}, 1920, 1080)
    clearly_better = attention.update(0.2, {7: (CENTER, LOOKING), 8: (closer, AWAY)}, 1920, 1080)

    assert first is not None and (first.track_id, first.eye_contact) == (7, True)
    assert slightly_better is not None and slightly_better.track_id == 7
    assert clearly_better is not None
    assert (clearly_better.track_id, clearly_better.eye_contact) == (8, False)
    assert clearly_better.gaze_off_camera == pytest.approx(40.0)
    assert attention.update(0.3, {}, 1920, 1080) is None
