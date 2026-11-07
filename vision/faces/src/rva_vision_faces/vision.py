import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
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
from rva_schemas.observation import VisualObservation
from rva_schemas.record import CameraFrame, DepthFrame, LiveFrame, PeopleInView
from rva_schemas.vision import (
    Body,
    CameraIntrinsics,
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
from rva_vision_faces.body import POSE_ITEM, BodyEstimator, SteadyHands, attach_seen_first
from rva_vision_faces.camera import Camera
from rva_vision_faces.config import FaceVisionConfig
from rva_vision_faces.database import FaceDatabase
from rva_vision_faces.depth import DepthEstimator
from rva_vision_faces.detector import Detection, Image, YunetDetector
from rva_vision_faces.embedder import Embedding, SfaceEmbedder
from rva_vision_faces.face_features import EXPRESSION_ITEM, MESH_ITEM, FaceAnalyzer
from rva_vision_faces.gallery import Gallery
from rva_vision_faces.models import load_session
from rva_vision_faces.scene import Scene, people_in_view
from rva_vision_faces.space import Lifter
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
        camera = config.camera
        self.intrinsics: Final = (
            None
            if camera.intrinsics is None
            else load_yaml(CameraIntrinsics, camera.intrinsics).scaled_to(
                camera.width, camera.height
            )
        )
        """The camera's at the frame's resolution."""
        self.detector: Final = YunetDetector(
            load_session(config.detector.model, device, dynamic_image_size=True, batch_item=None),
            config.detector,
        )
        self.embedder: Final = SfaceEmbedder(
            load_session(config.recognizer, device, dynamic_image_size=False, batch_item=None)
        )
        features = config.face_features
        self.faces: Final = (
            None
            if features is None
            else FaceAnalyzer(
                *(
                    load_session(model, device, dynamic_image_size=False, batch_item=item)
                    for model, item in (
                        (features.landmarks, MESH_ITEM),
                        (features.blendshapes, None),
                        (features.gaze, None),
                        (features.expression, EXPRESSION_ITEM),
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
                load_session(body.detector, device, dynamic_image_size=False, batch_item=None),
                load_session(body.pose, device, dynamic_image_size=False, batch_item=POSE_ITEM),
                body,
            )
        )
        self.steady_hands: Final = SteadyHands()
        """Used in the frame thread only."""
        self.depth: Final = (
            None
            if config.depth is None
            else DepthEstimator(
                load_session(config.depth.model, device, dynamic_image_size=False, batch_item=None),
                self.intrinsics,
            )
        )
        self.lifter: Final = (
            None if self.depth is None or self.intrinsics is None else Lifter(self.intrinsics)
        )
        """Places what is seen in 3D, keeping each person's size: used in the frame thread only."""


class FaceVision:
    """`Vision` from a camera: detects, tracks and recognizes faces in a thread of its own."""

    def __init__(
        self,
        config: FaceVisionConfig,
        database: FaceDatabase,
        *,
        recorder: Recorder,
        record_video: bool,
        speaker: SpeakerDetector | None,
    ) -> None:
        """Set up the pipeline; the camera and models open when `run` starts.

        Args:
            config: Camera, models and thresholds.
            database: Known people.
            recorder: Receives, from the camera's thread, the people tracked in each frame, the
                frame itself as a `LiveFrame` for whatever shows the camera, and with
                `record_video` the frame compressed and its depth.
            record_video: Record the camera frames, scaled down and compressed, and their depth, to
                replay them.
            speaker: Tells which face speaks, with the microphone that feeds it; `None` without
                one.
        """
        self._config: Final = config
        self._database: Final = database
        self._recorder: Final = recorder
        self._record_video: Final = record_video
        # The user's last sentence, the track of the face in front when it was said, and when.
        self._caption: tuple[int, str, float] | None = None
        self._speaker: Final = speaker
        self._tracker: Final = Tracker(config.tracker)
        self._gallery = Gallery((), config.matching)
        self._granted_operators: frozenset[UUID] = frozenset()
        # Guards the tracker, which the frame thread changes and the event loop reads.
        self._lock: Final = threading.Lock()
        self._subscribers: Final[set[asyncio.Queue[FaceEvent]]] = set()
        self._observers: Final[set[asyncio.Queue[VisualObservation]]] = set()

    async def run(self) -> None:
        """Process frames until cancelled."""
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

    @asynccontextmanager
    async def observations(self) -> AsyncGenerator[AsyncIterator[VisualObservation]]:
        """Subscribe to what is seen in each frame, for the visual attention.

        A subscriber that falls behind gets the latest frame, not every one: attention is about
        now.
        """
        queue = asyncio.Queue[VisualObservation](maxsize=1)
        self._observers.add(queue)
        try:
            yield _drain(queue)
        finally:
            self._observers.discard(queue)

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
        attention = (
            None
            if config.face_features is None
            else Attention(config.face_features.eye_contact, config.face_features.lip_motion)
        )
        logger.info("Watching {} on {}", config.camera.device, config.device.kind)
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
                frame, scene, depth, main_user, captured_at, frame_timings = self._step(
                    camera, models, attention, loop
                )
                now = time.perf_counter()
                timings = _smooth(
                    timings, frame_timings.model_copy(update={"fps": 1 / (now - previous)})
                )
                previous = now
                if now - last_logged >= _LOG_EVERY_S:
                    logger.debug("Vision timings: {}", timings)
                    last_logged = now
                people = self._record(frame, scene, depth, main_user, timings, models.intrinsics)
                if self._observers:
                    observation = VisualObservation(
                        time=captured_at,
                        people=people,
                        main_user=None if main_user is None else main_user.track_id,
                    )
                    loop.call_soon_threadsafe(self._observe, observation)
        finally:
            models.pool.shutdown()
            camera.close()

    def _step(
        self,
        camera: Camera,
        models: _Models,
        attention: Attention | None,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[Image, Scene, DepthMap | None, MainUser | None, float, VisionTimings]:
        """Process one frame.

        Returns the image, what was seen, its depth, the main user, when the frame was taken on
        the monotonic clock, and the stages' times.
        """
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
            seen_ids = {track_id for track_id, _ in visible}
            hidden = [
                (track.track_id, track.box)
                for track in self._tracker.tracks
                if track.track_id not in seen_ids
            ]
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
            bodies, unmatched = attach_seen_first(
                found.result(), [(id_, d.box) for id_, d in visible], hidden
            )
            bodies = models.steady_hands.filter(bodies)
        posed = time.perf_counter()
        depth = None if measured is None else measured.result()
        sounded = time.perf_counter()
        speakers = {} if heard is None else heard.result()
        listened = time.perf_counter()
        scene = Scene(
            faces=faces, bodies=bodies, unmatched_bodies=tuple(unmatched), speakers=speakers
        )
        main_user = None if attention is None else self._attend(attention, frame, scene)
        if depth is not None and models.lifter is not None:
            with _tracer.start_as_current_span("vision:lift"):
                scene = models.lifter.lift(
                    scene, depth, None if main_user is None else main_user.track_id
                )
        return (
            frame,
            scene,
            depth,
            main_user,
            captured_at,
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

    def _record(
        self,
        frame: Image,
        scene: Scene,
        depth: DepthMap | None,
        main_user: MainUser | None,
        timings: VisionTimings,
        intrinsics: CameraIntrinsics | None,
    ) -> PeopleInView:
        """Record what was seen, the frame live, and with `record_video` the frame compressed.

        Returns the people in the frame's pixels; in a recording with video, they are kept in the
        recorded video's pixels instead, their points in 3D in metres either way.
        """
        height, width = frame.shape[:2]
        caption = self._current_caption()
        people = people_in_view(
            self._tracker.tracks, scene, 1.0, width, height, intrinsics, caption
        )
        if not self._record_video:
            self._recorder.record(people)
        else:
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
            if depth is not None and intrinsics is not None:
                rows, columns = depth.metres.shape
                self._recorder.record(
                    DepthFrame(depth=depth, intrinsics=intrinsics.scaled_to(columns, rows))
                )
            self._recorder.record(
                people_in_view(
                    self._tracker.tracks,
                    scene,
                    scale,
                    round(width * scale),
                    round(height * scale),
                    intrinsics,
                    caption,
                )
            )
        # Last, once the frame is encoded: whatever shows it draws over it.
        self._recorder.record(
            LiveFrame(frame=frame, depth=depth, people=people, main_user=main_user, timings=timings)
        )
        return people

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

    def _observe(self, observation: VisualObservation) -> None:
        """Hand the frame's observation to each observer, replacing one they have not taken."""
        for queue in self._observers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(observation)


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


async def _drain[T](queue: asyncio.Queue[T]) -> AsyncIterator[T]:
    while True:
        yield await queue.get()
