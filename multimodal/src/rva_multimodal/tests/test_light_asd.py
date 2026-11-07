from collections import deque
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from rva_multimodal.config import ActiveSpeakerConfig
from rva_multimodal.light_asd import LightAsd, at_frame_rate, face_crop, mfcc
from rva_schemas.vision import ActiveSpeaker, BoundingBox

CONFIG = ActiveSpeakerConfig(
    device="cpu",
    model_dir=Path("/nonexistent"),
    window_s=0.2,
    delay_s=0.08,
    enter=-1.0,
    leave=-1.3,
)
"""Windows of 5 frames, scored 2 frames before the newest."""
FRAME = np.zeros((720, 1280, 3), np.uint8)
FACE = BoundingBox(x=600, y=300, width=100, height=120)


class FakeLightAsd:
    """Scores the frame 2 before the newest with the next value it is given.

    The other frames score nonsense; the shapes of the inputs are kept.
    """

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.inputs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(
        self, sound: npt.NDArray[np.float32], faces: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        self.inputs.append((sound.shape, faces.shape))
        scores = np.full(faces.shape[1], 99.0, np.float32)
        scores[max(faces.shape[1] - 3, 0)] = self.scores.pop(0)
        return scores


def test_the_crop_is_the_lower_middle_of_a_square_around_the_face_padded_beyond_the_frame() -> None:
    gray = np.zeros((720, 1280), np.uint8)
    # Around the middle of the crop: the face's middle, as the crop reaches lower than the box.
    gray[350:450, 600:700] = 255

    middle = face_crop(gray, FACE)
    at_the_corner = face_crop(gray, BoundingBox(x=-50, y=-60, width=100, height=120))

    assert middle.shape == (112, 112)
    assert middle[56, 56] == 255
    assert at_the_corner.shape == (112, 112)
    assert at_the_corner[0, 0] == 110


def test_mfcc_gives_four_frames_per_video_frame_of_sound_and_its_window() -> None:
    sound = np.random.default_rng(0).normal(0, 0.1, 10 * 640 + 240).astype(np.float32)

    assert mfcc(sound).shape == (40, 13)


def test_crops_are_resampled_to_25_fps_repeating_the_closest_in_time() -> None:
    history = deque((time, np.full((112, 112), value, np.uint8)) for time, value in (
        (0.04, 1), (0.09, 2),
    ))  # fmt: skip

    crops = at_frame_rate(history, 0.09, 3)

    assert [crop[0, 0] for crop in crops] == [1, 1, 2]


def test_a_face_is_scored_once_enough_sound_is_heard_and_speaks_with_hysteresis() -> None:
    scorer = FakeLightAsd([-0.5, -1.25, -1.5])
    speaker = LightAsd(CONFIG, scorer)

    unheard = speaker.see(0.0, FRAME, {3: FACE})
    speaker.hear(1.0, np.zeros(16000, np.float32))
    seen = [speaker.see(time, FRAME, {3: FACE}) for time in (0.92, 0.96, 1.0)]

    assert unheard == {}
    assert seen == [
        {3: ActiveSpeaker(score=-0.5, speaking=True)},
        {3: ActiveSpeaker(score=-1.25, speaking=True)},
        {3: ActiveSpeaker(score=-1.5, speaking=False)},
    ]
    assert scorer.inputs == [((1, 4, 13), (1, 1, 112, 112)), ((1, 8, 13), (1, 2, 112, 112)), (
        (1, 12, 13), (1, 3, 112, 112),
    )]  # fmt: skip


def test_a_face_that_leaves_starts_over_when_it_comes_back() -> None:
    scorer = FakeLightAsd([-0.5, -1.25])
    speaker = LightAsd(CONFIG, scorer)
    speaker.hear(1.0, np.zeros(16000, np.float32))

    speaker.see(0.9, FRAME, {3: FACE})
    speaker.see(0.95, FRAME, {})
    back = speaker.see(1.0, FRAME, {3: FACE})

    assert back == {3: ActiveSpeaker(score=-1.25, speaking=False)}
    assert scorer.inputs[-1] == ((1, 4, 13), (1, 1, 112, 112))


def test_the_config_refuses_to_stop_speaking_above_where_it_starts() -> None:
    with pytest.raises(ValueError, match="leave -0.5 is above enter -1.0"):
        ActiveSpeakerConfig.model_validate(CONFIG.model_dump() | {"leave": -0.5})
