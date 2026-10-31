import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Final
from uuid import UUID

import cv2
from loguru import logger
from opentelemetry import trace

from rva_schemas.abc.recorder import Recorder
from rva_schemas.record import CameraFrame, FaceBox, FaceBoxes
from rva_schemas.vision import (
    BoundingBox,
    FaceEvent,
    KnownPerson,
    NotRecognizedError,
    Remembered,
    TrackedFace,
    UnknownTrackError,
)
from rva_vision_faces.camera import Camera
from rva_vision_faces.config import FaceVisionConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.detector import Detection, Image, YunetDetector
from rva_vision_faces.embedder import Embedding, SfaceEmbedder
from rva_vision_faces.gallery import Gallery
from rva_vision_faces.models import load_session
from rva_vision_faces.tracker import Track, Tracker
from rva_vision_faces.window import DebugWindow, Timings, draw

_tracer: Final = trace.get_tracer("rva_vision_faces")
_WINDOW_SIZE: Final = (960, 540)
_SMOOTHING: Final = 0.1
_LOG_EVERY_S: Final = 5.0
_CAPTION_S: Final = 6.0
"""How long the user's last sentence stays over the image."""


class FaceVision:
    """`Vision` from a camera: detects, tracks and recognizes faces in a thread of its own."""

    def __init__(
        self,
        config: FaceVisionConfig,
        database: FaceDatabase,
        *,
        recorder: Recorder,
        record_video: bool,
        show_window: bool,
        on_key: Callable[[str], None] | None,
    ) -> None:
        """Set up the pipeline; the camera and models open when `run` starts.

        Args:
            config: Camera, models and thresholds.
            database: Known people.
            recorder: Receives the faces tracked in each frame, and the frames themselves with
                `record_video`.
            record_video: Record the camera frames, scaled down and compressed, to replay them.
            show_window: Show the camera with the tracked faces in a window.
            on_key: Called in the event loop with each key pressed in the window, except `q`.
        """
        self._config: Final = config
        self._database: Final = database
        self._recorder: Final = recorder
        self._record_video: Final = record_video
        self._show_window: Final = show_window
        # The user's last sentence and when it was said, on the monotonic clock.
        self._caption: tuple[str, float] | None = None
        self._on_key: Final = on_key
        self._tracker: Final = Tracker(config.tracker)
        self._gallery = Gallery((), config.matching)
        self._granted_operators: frozenset[UUID] = frozenset()
        # Guards the tracker, which the frame thread changes and the event loop reads.
        self._lock: Final = threading.Lock()
        self._subscribers: Final[set[asyncio.Queue[FaceEvent]]] = set()

    async def run(self) -> None:
        """Process frames until cancelled, or until the window is closed or `q` is pressed."""
        loop = asyncio.get_running_loop()
        await self._reload()
        stop = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._process, loop, stop))
        try:
            await asyncio.shield(worker)
        finally:
            stop.set()
            await worker

    @asynccontextmanager
    async def events(self) -> AsyncGenerator[AsyncIterator[FaceEvent]]:
        queue = asyncio.Queue[FaceEvent]()
        self._subscribers.add(queue)
        try:
            yield _drain(queue)
        finally:
            self._subscribers.discard(queue)

    def faces(self) -> tuple[TrackedFace, ...]:
        with self._lock:
            return self._tracker.faces()

    async def remember(self, track_id: int, name: str) -> Remembered:
        with self._lock:
            track = self._tracker.find(track_id)
            if track is None:
                raise UnknownTrackError(track_id)
            embeddings = list(track.embeddings)
            previous = None if track.match is None else track.match.person
        person = await asyncio.to_thread(self._database.add, name, embeddings)
        await self._reload()
        logger.info("Remembered track {} as {} ({})", track_id, name, person.user_id)
        return Remembered(person=person, previous=None if previous == person else previous)

    async def forget(self, name: str) -> KnownPerson:
        person = await asyncio.to_thread(self._database.forget, name)
        await self._reload()
        logger.info("Forgot {} ({})", name, person.user_id)
        return person

    async def rename(self, track_id: int, name: str) -> KnownPerson:
        with self._lock:
            track = self._tracker.find(track_id)
            if track is None:
                raise UnknownTrackError(track_id)
            if track.match is None:
                raise NotRecognizedError(track_id)
            user_id = track.match.person.user_id
        person = await asyncio.to_thread(self._database.rename, user_id, name)
        await self._reload()
        logger.info("Renamed {} to {}", user_id, name)
        return person

    async def people(self) -> tuple[KnownPerson, ...]:
        return tuple(await asyncio.to_thread(self._database.people))

    def granted_operators(self) -> frozenset[UUID]:
        return self._granted_operators

    async def grant_operator(self, user_id: UUID, granted_by: UUID) -> None:
        await asyncio.to_thread(self._database.grant_operator, user_id, granted_by)
        await self._reload()
        logger.info("{} made {} an operator", granted_by, user_id)

    async def revoke_operator(self, user_id: UUID) -> bool:
        revoked = await asyncio.to_thread(self._database.revoke_operator, user_id)
        await self._reload()
        logger.info("Revoked operator {}: {}", user_id, revoked)
        return revoked

    def caption(self, text: str) -> None:
        """Show a sentence of the user's over the image for a few seconds."""
        self._caption = (text, time.monotonic())

    async def _reload(self) -> None:
        self._gallery = Gallery(await asyncio.to_thread(self._database.load), self._config.matching)
        self._granted_operators = await asyncio.to_thread(self._database.granted_operators)

    def _process(self, loop: asyncio.AbstractEventLoop, stop: threading.Event) -> None:
        cv2.setNumThreads(2)
        config = self._config
        detector = YunetDetector(
            load_session(config.detector.model, config.device, dynamic_image_size=True),
            config.detector,
        )
        embedder = SfaceEmbedder(
            load_session(config.recognizer, config.device, dynamic_image_size=False)
        )
        camera = Camera(config.camera)
        window = DebugWindow(*_WINDOW_SIZE) if self._show_window else None
        logger.info("Watching {} on {}", config.camera.device, config.device)
        timings = Timings(fps=0.0, capture_ms=0.0, detect_ms=0.0, embed_ms=0.0)
        previous = last_logged = time.perf_counter()
        try:
            while not stop.is_set():
                frame, frame_timings = self._step(camera, detector, embedder, loop)
                now = time.perf_counter()
                timings = _smooth(timings, frame_timings, 1 / (now - previous))
                previous = now
                if now - last_logged >= _LOG_EVERY_S:
                    logger.debug("Vision timings: {}", timings)
                    last_logged = now
                self._record(frame)
                if window is None:
                    continue
                draw(frame, self._tracker.tracks, timings, self._current_caption())
                key = window.show(frame)
                if key == "q":
                    return
                if key is not None and self._on_key is not None:
                    loop.call_soon_threadsafe(self._on_key, key)
        finally:
            camera.close()
            if window is not None:
                window.close()

    def _step(
        self,
        camera: Camera,
        detector: YunetDetector,
        embedder: SfaceEmbedder,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[Image, tuple[float, float, float]]:
        """Process one frame; returns it with its capture, detection and embedding times in s."""
        start = time.perf_counter()
        with _tracer.start_as_current_span("vision:capture"):
            frame = camera.read()
        captured = time.perf_counter()
        with _tracer.start_as_current_span("vision:detect"):
            min_height = self._config.tracker.min_face_height * frame.shape[0]
            detections = [d for d in detector.detect(frame) if d.box.height >= min_height]
        detected = time.perf_counter()
        embedding_time = 0.0

        def embed(detection: Detection) -> Embedding:
            nonlocal embedding_time
            begin = time.perf_counter()
            with _tracer.start_as_current_span("vision:embed"):
                embedding = embedder.embed(frame, detection)
            embedding_time += time.perf_counter() - begin
            return embedding

        with self._lock:
            events = self._tracker.update(time.monotonic(), detections, embed, self._gallery)
        for event in events:
            logger.info("Face event: {}", event)
        if events:
            loop.call_soon_threadsafe(self._publish, events)
        return frame, (captured - start, detected - captured, embedding_time)

    def _record(self, frame: Image) -> None:
        height, width = frame.shape[:2]
        scale = 1.0
        if self._record_video:
            video = self._config.video
            scale = video.width / width
            small = cv2.resize(
                frame, (video.width, round(height * scale)), interpolation=cv2.INTER_AREA
            )
            encoded, jpeg = cv2.imencode(
                ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, video.jpeg_quality]
            )
            if not encoded:
                raise RuntimeError("Could not encode a camera frame as JPEG")
            self._recorder.record(CameraFrame(jpeg=jpeg.tobytes()))
        self._recorder.record(
            face_boxes(self._tracker.tracks, scale, round(width * scale), round(height * scale))
        )

    def _current_caption(self) -> str | None:
        caption = self._caption
        if caption is None or time.monotonic() - caption[1] > _CAPTION_S:
            return None
        return caption[0]

    def _publish(self, events: Sequence[FaceEvent]) -> None:
        for queue in self._subscribers:
            for event in events:
                queue.put_nowait(event)


def face_boxes(tracks: Sequence[Track], scale: float, width: int, height: int) -> FaceBoxes:
    """The tracks' boxes in an image scaled by `scale`, of size `width` by `height`."""
    return FaceBoxes(
        image_width=width,
        image_height=height,
        boxes=tuple(
            FaceBox(
                track_id=track.track_id,
                box=BoundingBox(
                    x=track.box.x * scale,
                    y=track.box.y * scale,
                    width=track.box.width * scale,
                    height=track.box.height * scale,
                ),
                person=None if track.match is None else track.match.person,
                confirmed=track.confirmed,
            )
            for track in tracks
        ),
    )


def _smooth(timings: Timings, frame: tuple[float, float, float], fps: float) -> Timings:
    capture, detect, embed = frame

    def ema(old: float, new: float) -> float:
        return old + _SMOOTHING * (new - old) if old else new

    return Timings(
        fps=ema(timings.fps, fps),
        capture_ms=ema(timings.capture_ms, capture * 1e3),
        detect_ms=ema(timings.detect_ms, detect * 1e3),
        embed_ms=ema(timings.embed_ms, embed * 1e3),
    )


async def _drain(queue: asyncio.Queue[FaceEvent]) -> AsyncIterator[FaceEvent]:
    while True:
        yield await queue.get()
