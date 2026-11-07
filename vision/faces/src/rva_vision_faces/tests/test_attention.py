import pytest

from rva_schemas.vision import ActiveSpeaker, BoundingBox, FaceFeatures, Gaze
from rva_vision_faces.attention import Attention, lip_aperture, off_camera, prominence
from rva_vision_faces.config import EyeContactConfig, LipMotionConfig

EYE_CONTACT = EyeContactConfig(window_s=0.4, enter_deg=12.0, leave_deg=18.0)
LIP_MOTION = LipMotionConfig(window_s=0.4, enter=0.03, leave=0.015)
CENTER = BoundingBox(x=900, y=300, width=120, height=150)
"""A face in the middle of a 1920 x 1080 frame."""
LOOKING = Gaze(yaw=3.0, pitch=-2.0)
AWAY = Gaze(yaw=40.0, pitch=0.0)


def face(gaze: Gaze, aperture: float = 0.0) -> FaceFeatures:
    """A face whose lips are `aperture` times its mouth's width of 50 pixels apart."""
    landmarks = [(0.0, 0.0)] * 478
    landmarks[78], landmarks[308] = (475.0, 400.0), (525.0, 400.0)
    landmarks[13], landmarks[14] = (500.0, 400.0), (500.0, 400.0 + 50 * aperture)
    return FaceFeatures(
        landmarks=tuple(landmarks),
        left_eye_open=1.0,
        right_eye_open=1.0,
        mouth_open=0.0,
        gaze=gaze,
        gaze_ray=None,
        expression="neutral",
        expression_score=1.0,
    )


def test_off_camera_is_the_angle_from_looking_straight_at_it() -> None:
    assert off_camera(Gaze(yaw=0.0, pitch=0.0)) == 0.0
    assert off_camera(Gaze(yaw=-15.0, pitch=0.0)) == pytest.approx(15.0)
    assert off_camera(Gaze(yaw=0.0, pitch=20.0)) == pytest.approx(20.0)


def test_lip_aperture_is_the_gap_between_the_lips_over_the_mouths_width() -> None:
    assert lip_aperture(face(LOOKING).landmarks) == 0.0
    assert lip_aperture(face(LOOKING, 0.3).landmarks) == pytest.approx(0.3)


def test_prominence_favors_a_central_face_over_a_larger_one_at_the_edge() -> None:
    edge = BoundingBox(x=1650, y=300, width=160, height=200)

    assert prominence(CENTER, 1920, 1080) == pytest.approx(150 / 1080)
    assert prominence(edge, 1920, 1080) == pytest.approx(200 / 1080 * (1 - 770 / 960))
    assert prominence(BoundingBox(x=1900, y=0, width=200, height=200), 1920, 1080) == 0.0


def test_eye_contact_holds_through_a_blink_and_ends_when_the_gaze_turns_away() -> None:
    attention = Attention(EYE_CONTACT, LIP_MOTION)
    gazes = [LOOKING, LOOKING, LOOKING, AWAY, LOOKING, AWAY, AWAY, AWAY, AWAY]

    contact = [
        attention.update(0.1 * frame, {7: (CENTER, face(gaze))}, {}, 1920, 1080)
        for frame, gaze in enumerate(gazes)
    ]

    assert [user.eye_contact if user else None for user in contact] == [
        True, True, True, True, True, True, False, False, False,
    ]  # fmt: skip


def test_eye_contact_starts_within_the_tighter_threshold_and_ends_beyond_the_looser() -> None:
    attention = Attention(EYE_CONTACT.model_copy(update={"window_s": 0.0}), LIP_MOTION)

    def look(time: float, yaw: float) -> bool:
        user = attention.update(time, {7: (CENTER, face(Gaze(yaw=yaw, pitch=0.0)))}, {}, 1920, 1080)
        assert user is not None
        return user.eye_contact

    assert [
        look(0.1 * frame, yaw) for frame, yaw in enumerate((15.0, 10.0, 15.0, 17.0, 19.0, 15.0))
    ] == [
        False, True, True, True, False, False,
    ]  # fmt: skip


def test_the_lips_must_move_not_just_be_open() -> None:
    attention = Attention(EYE_CONTACT, LIP_MOTION)
    # Closed, then held wide open, then talking, then closed again; a frame every 0.1 s.
    apertures = [0.0] * 4 + [0.3] * 6 + [0.05, 0.2, 0.0, 0.15, 0.05, 0.2] + [0.0] * 6

    moving = []
    for frame, aperture in enumerate(apertures):
        user = attention.update(0.1 * frame, {7: (CENTER, face(LOOKING, aperture))}, {}, 1920, 1080)
        assert user is not None
        moving.append(user.lips_moving)

    assert moving == [
        False, False, False, False,
        True, True, True, True, False, False,  # opening moves the lips, holding them open does not
        True, True, True, True, True, True,
        True, True, True, True, False, False,  # the window bridges the end of a word
    ]  # fmt: skip


def test_the_main_user_stays_until_someone_clearly_more_prominent_comes() -> None:
    attention = Attention(EYE_CONTACT, LIP_MOTION)
    beside = BoundingBox(x=700, y=300, width=120, height=160)
    closer = BoundingBox(x=700, y=300, width=200, height=260)
    looking, away = face(LOOKING), face(AWAY)

    first = attention.update(0.0, {7: (CENTER, looking)}, {}, 1920, 1080)
    slightly_better = attention.update(
        0.1, {7: (CENTER, looking), 8: (beside, away)}, {}, 1920, 1080
    )
    clearly_better = attention.update(
        0.2, {7: (CENTER, looking), 8: (closer, away)}, {}, 1920, 1080
    )

    assert first is not None and (first.track_id, first.eye_contact) == (7, True)
    assert slightly_better is not None and slightly_better.track_id == 7
    assert clearly_better is not None
    assert (clearly_better.track_id, clearly_better.eye_contact) == (8, False)
    assert clearly_better.gaze_off_camera == pytest.approx(40.0)
    assert attention.update(0.3, {}, {}, 1920, 1080) is None


def test_whether_the_main_user_speaks_comes_from_the_speaker_detector() -> None:
    attention = Attention(EYE_CONTACT, LIP_MOTION)
    speaking = ActiveSpeaker(score=0.4, speaking=True)
    beside = BoundingBox(x=100, y=300, width=120, height=150)

    heard = attention.update(
        0.0, {7: (CENTER, face(LOOKING)), 8: (beside, face(LOOKING))}, {8: speaking}, 1920, 1080
    )
    unheard = attention.update(0.1, {7: (CENTER, face(LOOKING))}, {}, 1920, 1080)

    assert heard is not None and (heard.track_id, heard.active_speaker) == (7, None)
    assert unheard is not None and unheard.active_speaker is None
    main = attention.update(0.2, {7: (CENTER, face(LOOKING))}, {7: speaking}, 1920, 1080)
    assert main is not None and main.active_speaker == speaking
