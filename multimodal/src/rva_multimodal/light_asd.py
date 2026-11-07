import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime
import python_speech_features
from loguru import logger

from rva_multimodal.config import ActiveSpeakerConfig
from rva_schemas.vision import ActiveSpeaker, BoundingBox

CHECKPOINT_URL: Final = (
    "https://github.com/Junhua-Liao/Light-ASD/raw/ed38c232de5efe0261dbd68627c0ade7cdfe14eb"
    "/weight/pretrain_AVA_CVPR.model"
)
"""Light-ASD trained on AVA-ActiveSpeaker, MIT: pinned, since the thresholds are tuned to it."""
CHECKPOINT_SHA256: Final = "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317"
_ONNX_FILE: Final = f"light_asd_ava_{CHECKPOINT_SHA256[:12]}.onnx"

FPS: Final = 25
"""The video frame rate Light-ASD was trained on, with 4 MFCC frames of 10 ms each."""
SAMPLE_RATE: Final = 16000
_SAMPLES_PER_FRAME: Final = SAMPLE_RATE // FPS
_MFCC_WINDOW: Final = 400
_MFCC_STEP: Final = 160
_CROP_SCALE: Final = 0.4
"""How far below and beside the face box the crop reaches, as in Light-ASD's own demo."""
_CROP_SIZE: Final = 112
_PADDING: Final = 110
"""Gray level of the crop beyond the frame's edges."""

type Scorer = Callable[[npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]]
"""Light-ASD: MFCC of shape (1, 4T, 13) and faces of shape (1, T, 112, 112) to a logit per frame."""


class LightAsd:
    """`SpeakerDetector` with Light-ASD: whether each face's lips go with the sound heard."""

    def __init__(self, config: ActiveSpeakerConfig, scorer: Scorer) -> None:
        self._config: Final = config
        self._scorer: Final = scorer
        self._frames: Final = round(config.window_s * FPS)
        self._delay: Final = round(config.delay_s * FPS)
        # The microphone's thread writes the sound, the camera's reads it.
        self._lock: Final = threading.Lock()
        self._sound: Final = deque[npt.NDArray[np.float32]]()
        self._sound_samples = 0
        self._heard_at = 0.0
        """When the last sample was heard."""
        self._faces: Final[dict[int, deque[tuple[float, npt.NDArray[np.uint8]]]]] = {}
        self._speaking: Final[dict[int, bool]] = {}

    def hear(self, time: float, samples: npt.NDArray[np.float32]) -> None:
        # A window's worth, and a second more: frames reach `see` after the sound of their time.
        kept = self._frames * _SAMPLES_PER_FRAME + _MFCC_WINDOW + SAMPLE_RATE
        with self._lock:
            self._sound.append(samples)
            self._sound_samples += len(samples)
            self._heard_at = time
            while self._sound_samples - len(self._sound[0]) >= kept:
                self._sound_samples -= len(self._sound.popleft())

    def see(
        self, time: float, frame: npt.NDArray[np.uint8], faces: Mapping[int, BoundingBox]
    ) -> dict[int, ActiveSpeaker]:
        for track_id in self._faces.keys() - faces.keys():
            del self._faces[track_id]
            self._speaking.pop(track_id, None)
        gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        speakers = {}
        for track_id, box in faces.items():
            history = self._faces.setdefault(track_id, deque())
            history.append((time, face_crop(gray, box)))
            while history[0][0] < time - self._config.window_s:
                history.popleft()
            count = min(self._frames, round((time - history[0][0]) * FPS) + 1)
            sound = self._sound_ending(time, count * _SAMPLES_PER_FRAME + _MFCC_WINDOW - _MFCC_STEP)
            if sound is None:
                continue
            scores = self._scorer(
                mfcc(sound)[np.newaxis], at_frame_rate(history, time, count)[np.newaxis]
            )
            score = float(scores[max(count - 1 - self._delay, 0)])
            threshold = (
                self._config.leave if self._speaking.get(track_id, False) else self._config.enter
            )
            self._speaking[track_id] = score > threshold
            speakers[track_id] = ActiveSpeaker(score=score, speaking=self._speaking[track_id])
        return speakers

    def _sound_ending(self, time: float, samples: int) -> npt.NDArray[np.float32] | None:
        """The last `samples` heard by `time`, or `None` if not that many were."""
        with self._lock:
            if self._sound_samples == 0:
                return None
            sound = np.concatenate(self._sound)
            heard_at = self._heard_at
        end = len(sound) - max(round((heard_at - time) * SAMPLE_RATE), 0)
        return None if end < samples else sound[end - samples : end]


def open_light_asd(config: ActiveSpeakerConfig) -> LightAsd:
    """Load Light-ASD on the config's device, converting its checkpoint to ONNX the first time.

    Raises:
        subprocess.CalledProcessError: If the conversion fails, e.g. without the `export` extra.
    """
    model = config.model_dir.expanduser() / _ONNX_FILE
    if not model.exists():
        logger.info("Converting Light-ASD to ONNX at {}, once", model)
        # PyTorch is only needed here, so the conversion runs in a process of its own.
        subprocess.run([sys.executable, "-m", "rva_multimodal.export", str(model)], check=True)
    match config.device:
        case "cuda":
            # Loads the CUDA and cuDNN libraries installed from PyPI by onnxruntime-gpu's extras.
            onnxruntime.preload_dlls()
            providers = ["CUDAExecutionProvider"]
        case "cpu":
            providers = ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(model), providers=providers)
    if session.get_providers()[0] != providers[0]:
        raise RuntimeError(f"Light-ASD could not run on {config.device}: {session.get_providers()}")

    def score(
        sound: npt.NDArray[np.float32], faces: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        (scores,) = session.run(None, {"audio": sound, "faces": faces})
        return np.asarray(scores, dtype=np.float32)

    return LightAsd(config, score)


def face_crop(gray: npt.NDArray[np.uint8], box: BoundingBox) -> npt.NDArray[np.uint8]:
    """The 112x112 crop Light-ASD reads: the lower middle of a square around the face.

    As in Light-ASD's demo: a square of the face's larger side, plus 40% below and beside it,
    resized to 224x224, of which the middle 112x112 is kept.
    """
    half = max(box.width, box.height) / 2
    x, y = box.x + box.width / 2, box.y + box.height / 2
    top, bottom = round(y - half), round(y + half * (1 + 2 * _CROP_SCALE))
    left, right = round(x - half * (1 + _CROP_SCALE)), round(x + half * (1 + _CROP_SCALE))
    square = np.full((bottom - top, right - left), _PADDING, np.uint8)
    height, width = gray.shape
    inside_top, inside_bottom = max(top, 0), min(bottom, height)
    inside_left, inside_right = max(left, 0), min(right, width)
    if inside_top < inside_bottom and inside_left < inside_right:
        square[inside_top - top : inside_bottom - top, inside_left - left : inside_right - left] = (
            gray[inside_top:inside_bottom, inside_left:inside_right]
        )
    resized = np.asarray(cv2.resize(square, (2 * _CROP_SIZE, 2 * _CROP_SIZE)), dtype=np.uint8)
    start = _CROP_SIZE // 2
    return resized[start : start + _CROP_SIZE, start : start + _CROP_SIZE]


def at_frame_rate(
    history: deque[tuple[float, npt.NDArray[np.uint8]]], time: float, count: int
) -> npt.NDArray[np.float32]:
    """`count` crops at Light-ASD's frame rate up to `time`, each the one seen closest to it."""
    seen = np.array([seen_at for seen_at, _ in history])
    wanted = time - (count - 1 - np.arange(count)) / FPS
    nearest = np.abs(seen[np.newaxis] - wanted[:, np.newaxis]).argmin(axis=1)
    return np.stack([history[index][1] for index in nearest]).astype(np.float32)


def mfcc(sound: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """13 MFCC every 10 ms over 25 ms, as Light-ASD was trained on: 4 per video frame.

    `python_speech_features` takes the samples at 16-bit scale, which its energy term depends on.
    """
    return python_speech_features.mfcc(
        sound * 32768,
        SAMPLE_RATE,
        numcep=13,
        winlen=_MFCC_WINDOW / SAMPLE_RATE,
        winstep=_MFCC_STEP / SAMPLE_RATE,
    ).astype(np.float32)
