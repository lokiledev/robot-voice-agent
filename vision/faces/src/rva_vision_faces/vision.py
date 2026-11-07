import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Final
from uuid import UUID

import cv2
from loguru import logger
from opentelemetry import trace

from rva_pydantic import load_yaml
from rva_schemas.abc.recorder import Recorder
from rva_schemas.abc.speaker import SpeakerDetector
from rva_schemas.abc.vision import FrameViewer
from rva_schemas.record import CameraFrame
from rva_schemas.vision import (
    Body,
    DepthMap,
    FaceEvent,
    KnownPerson,
    MainUser,
    NotRecognizedError,
    Remembered,
    TrackedFace,
    UnknownTrackError,
    VisionTimings,
)
from rva_vision_faces.attention import Attention
from rva_vision_faces.body import BodyEstimator, attach
from rva_vision_faces.camera import Camera
from rva_vision_faces.config import CameraIntrinsics, FaceVisionConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.depth import DepthEstimator
from rva_vision_faces.detector import Detection, Image, YunetDetector
from rva_vision_faces.embedder import Embedding, SfaceEmbedder
from rva_vision_faces.face_features import FaceAnalyzer
from rva_vision_faces.gallery import Gallery
from rva_vision_faces.models import load_session
from rva_vision_faces.overlay import draw
from rva_vision_faces.scene import Scene, people_in_view
from rva_vision_faces.stream import RtspPublisher
from rva_vision_faces.tracker import Tracker

_tracer: Final = trace.get_tracer("rva_vision_faces")
_SMOOTHING: Final = 0.1
_LOG_EVERY_S: Final = 5.0
_CAPTION_S: Final = 6.0
"""How long the user's last sentence stays over the image."""


class _Models:
    """The models of the frame thread, loaded in it."""

    def __init__(self, config: FaceVisionConfig) -> None:
        device = config.device
        self.detector: Final = YunetDetector(
            load_session(config.detector.model, device, dynamic_image_size=True), config.detector
        )
        self.embedder: Final = SfaceEmbedder(
            load_session(config.recognizer, device, dynamic_image_size=False)
        )
        features = config.face_features
        self.faces: Final = (
            None
            if features is None
            else FaceAnalyzer(
                *(
                    load_session(model, device, dynamic_image_size=False)
                    for model in (
                        features.landmarks,
                        features.blendshapes,
                        features.gaze,
                        features.expression,
                    )
                )
            )
        )
        body = config.body
        self.pool: Final = ThreadPoolExecutor(3, thread_name_prefix="vision-alongside")
        """Runs the skeletons, the depth and who speaks alongside the faces: ONNX Runtime lets go
        of the GIL."""
        self.bodies: Final = (
            None
            if body is None
            else BodyEstimator(
                load_session(body.detector, device, dynamic_image_size=False),
                load_session(body.pose, device, dynamic_image_size=False),
                body,
            )
        )
        self.depth: Final = (
            None
            if config.depth is None
            else DepthEstimator(
                load_session(config.depth.model, device, dynamic_image_size=False),
                None
                if config.camera.intrinsics is None
                else load_yaml(CameraIntrinsics, config.camera.intrinsics),
            )
        )


class FaceVision:
    """`Vision` from a camera: detects, tracks and recognizes faces in a thread of its own."""

    def __init__(
        self,
        config: FaceVisionConfig,
        database: FaceDatabase,
        *,
        recorder: Recorder,
        record_video: bool,
        stream_video: bool,
        viewer: Callable[[], FrameViewer] | None,
        on_key: Callable[[str], None] | None,
        speaker: SpeakerDetector | None,
    ) -> None:
        """Set up the pipeline; the camera and models open when `run` starts.

        Args:
            config: Camera, models and thresholds.
            database: Known people.
            recorder: Receives the faces tracked in each frame, and the frames themselves with
                `record_video`.
            record_video: Record the camera frames, scaled down and compressed, to replay them.
            stream_video: Publish the camera with the tracked faces to the RTSP server of the
                config, e.g. for a web page to show it.
            viewer: Opens a viewer to show the camera with the tracked people, for debugging;
                called in the camera's thread, which the viewer then lives in. `None` to run
                without one.
            on_key: Called in the event loop with each key pressed in the viewer, except `q`.
            speaker: Tells which face speaks, with the microphone that feeds it; `None` without
                one.
        """
        self._config: Final = config
        self._database: Final = database
        self._recorder: Final = recorder
        self._record_video: Final = record_video
        self._stream_video: Final = stream_video
        self._viewer: Final = viewer
        # The user's last sentence, the track of the face in front when it was said, and when.
        self._caption: tuple[int, str, float] | None = None
        self._on_key: Final = on_key
        self._speaker: Final = speaker
        self._tracker: Final = Tracker(config.tracker)
        self._gallery = Gallery((), config.matching)
        self._granted_operators: frozenset[UUID] = frozenset()
        # Guards the tracker, which the frame thread changes and the event loop reads.
        self._lock: Final = threading.Lock()
        self._subscribers: Final[set[asyncio.Queue[FaceEvent]]] = set()

    async def run(self) -> None:
        """Process frames until cancelled, or until the viewer is closed or `q` is pressed."""
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
        """Show a sentence of the user's on their face's box for a few seconds.

        It goes to the face in front of the robot, taken as the speaker, if there is one.
        """
        nearest = next((face for face in self.faces() if face.nearest), None)
        if nearest is not None:
            self._caption = (nearest.track_id, text, time.monotonic())

    async def _reload(self) -> None:
        gallery = Gallery(await asyncio.to_thread(self._database.load), self._config.matching)
        with self._lock:
            self._gallery = gallery
            self._tracker.recognize(gallery)
        self._granted_operators = await asyncio.to_thread(self._database.granted_operators)

    def _process(self, loop: asyncio.AbstractEventLoop, stop: threading.Event) -> None:
        cv2.setNumThreads(2)
        config = self._config
        models = _Models(config)
        camera = Camera(config.camera)
        window = None if self._viewer is None else self._viewer()
        attention = (
            None
            if config.face_features is None
            else Attention(config.face_features.eye_contact, config.face_features.lip_motion)
        )
        publisher = (
            RtspPublisher(config.stream, config.camera.width, config.camera.height)
            if self._stream_video
            else None
        )
        logger.info("Watching {} on {}", config.camera.device, config.device)
        timings = VisionTimings(
            fps=0.0,
            capture_ms=0.0,
            detect_ms=0.0,
            embed_ms=0.0,
            faces_ms=0.0,
            bodies_ms=0.0,
            depth_ms=0.0,
            speaker_ms=0.0,
        )
        previous = last_logged = time.perf_counter()
        try:
            while not stop.is_set():
                frame, scene, depth, frame_timings = self._step(camera, models, loop)
                now = time.perf_counter()
                timings = _smooth(
                    timings, frame_timings.model_copy(update={"fps": 1 / (now - previous)})
                )
                previous = now
                main_user = None if attention is None else self._attend(attention, frame, scene)
                if now - last_logged >= _LOG_EVERY_S:
                    logger.debug("Vision timings: {}", timings)
                    last_logged = now
                self._record(frame, scene)
                if window is None and publisher is None:
                    continue
                draw(frame, self._tracker.tracks, scene, timings, self._current_caption())
                if publisher is not None:
                    publisher.send(frame)
                if window is None:
                    continue
                height, width = frame.shape[:2]
                people = people_in_view(
                    self._tracker.tracks, scene, 1.0, width, height, self._current_caption()
                )
                key = window.show(frame, people, depth, main_user, timings)
                if key == "q":
                    return
                if key is not None and self._on_key is not None:
                    loop.call_soon_threadsafe(self._on_key, key)
        finally:
            models.pool.shutdown()
            camera.close()
            if publisher is not None:
                publisher.close()
            if window is not None:
                window.close()

    def _step(
        self, camera: Camera, models: _Models, loop: asyncio.AbstractEventLoop
    ) -> tuple[Image, Scene, DepthMap | None, VisionTimings]:
        """Process one frame; returns it with what was seen, its depth and the stages' times."""
        start = time.perf_counter()
        with _tracer.start_as_current_span("vision:capture"):
            frame = camera.read()
        captured = time.perf_counter()
        captured_at = time.monotonic()
        # Skeletons need no face: they are estimated meanwhile, on the other thread.
        found = (
            None if models.bodies is None else models.pool.submit(_estimate, models.bodies, frame)
        )
        measured = (
            None if models.depth is None else models.pool.submit(_measure, models.depth, frame)
        )
        with _tracer.start_as_current_span("vision:detect"):
            min_height = self._config.tracker.min_face_height * frame.shape[0]
            detections = [d for d in models.detector.detect(frame) if d.box.height >= min_height]
        detected = time.perf_counter()
        embedding_time = 0.0

        def embed(detection: Detection) -> Embedding:
            nonlocal embedding_time
            begin = time.perf_counter()
            with _tracer.start_as_current_span("vision:embed"):
                embedding = models.embedder.embed(frame, detection)
            embedding_time += time.perf_counter() - begin
            return embedding

        with self._lock:
            events = self._tracker.update(time.monotonic(), detections, embed, self._gallery)
            visible = [(track.track_id, track.detection) for track in self._tracker.visible()]
        for event in events:
            logger.info("Face event: {}", event)
        if events:
            loop.call_soon_threadsafe(self._publish, events)
        # The sound is matched to the faces meanwhile too, once they are tracked.
        heard = (
            None
            if self._speaker is None
            else models.pool.submit(
                self._speaker.see, captured_at, frame, {id_: d.box for id_, d in visible}
            )
        )
        tracked = time.perf_counter()
        faces = {}
        if models.faces is not None:
            with _tracer.start_as_current_span("vision:faces"):
                features = models.faces.analyze(frame, [detection for _, detection in visible])
            faces = {track_id: f for (track_id, _), f in zip(visible, features, strict=True)}
        analyzed = time.perf_counter()
        bodies, unmatched = {}, []
        if found is not None:
            bodies, unmatched = attach(found.result(), [(id_, d.box) for id_, d in visible])
        posed = time.perf_counter()
        depth = None if measured is None else measured.result()
        sounded = time.perf_counter()
        speakers = {} if heard is None else heard.result()
        listened = time.perf_counter()
        scene = Scene(
            faces=faces, bodies=bodies, unmatched_bodies=tuple(unmatched), speakers=speakers
        )
        return (
            frame,
            scene,
            depth,
            VisionTimings(
                fps=0.0,
                capture_ms=(captured - start) * 1e3,
                detect_ms=(detected - captured) * 1e3,
                embed_ms=embedding_time * 1e3,
                faces_ms=(analyzed - tracked) * 1e3,
                bodies_ms=(posed - analyzed) * 1e3,
                depth_ms=(sounded - posed) * 1e3,
                speaker_ms=(listened - sounded) * 1e3,
            ),
        )

    def _attend(self, attention: Attention, frame: Image, scene: Scene) -> MainUser | None:
        boxes = {track.track_id: track.box for track in self._tracker.tracks}
        height, width = frame.shape[:2]
        return attention.update(
            time.monotonic(),
            {track_id: (boxes[track_id], face) for track_id, face in scene.faces.items()},
            scene.speakers,
            width,
            height,
        )

    def _record(self, frame: Image, scene: Scene) -> None:
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
            people_in_view(
                self._tracker.tracks,
                scene,
                scale,
                round(width * scale),
                round(height * scale),
                self._current_caption(),
            )
        )

    def _current_caption(self) -> tuple[int, str] | None:
        """The track and sentence to show, while still fresh."""
        caption = self._caption
        if caption is None or time.monotonic() - caption[2] > _CAPTION_S:
            return None
        return caption[0], caption[1]

    def _publish(self, events: Sequence[FaceEvent]) -> None:
        for queue in self._subscribers:
            for event in events:
                queue.put_nowait(event)


def _estimate(estimator: BodyEstimator, frame: Image) -> list[Body]:
    with _tracer.start_as_current_span("vision:bodies"):
        return estimator.estimate(frame)


def _measure(estimator: DepthEstimator, frame: Image) -> DepthMap:
    with _tracer.start_as_current_span("vision:depth"):
        return estimator.estimate(frame)


def _smooth(timings: VisionTimings, frame: VisionTimings) -> VisionTimings:
    def ema(old: float, new: float) -> float:
        return old + _SMOOTHING * (new - old) if old else new

    return VisionTimings(
        **{
            name: ema(getattr(timings, name), getattr(frame, name))
            for name in VisionTimings.model_fields
        }
    )


async def _drain(queue: asyncio.Queue[FaceEvent]) -> AsyncIterator[FaceEvent]:
    while True:
        yield await queue.get()
