"""`lune-sim`: the simulated Reachy Mini, with its people and what they say."""

import asyncio
import contextlib
import functools
import hashlib
import logging
import signal
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, assert_never

import cyclopts
import genesis as gs
import httpx
from genesis.constants import backend
from loguru import logger

from rva_app.config import RuntimeConfig
from rva_pydantic import load_yaml
from rva_robot_sim.actor import cast
from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.hdri import install_hdri
from rva_robot_sim.room import Room
from rva_robot_sim.scene import ReachyMiniScene
from rva_robot_sim.scenes import SCENES, SceneDescription, describe
from rva_robot_sim.serve import serve
from rva_robot_sim.speech import SpeechConfig, SpokenLine, bake
from rva_voice_tts.kokoro import KokoroConfig, KokoroSynthesizer

app = cyclopts.App(
    name="lune-sim",
    help="Simulate a Reachy Mini on a table, with people around, serving its joints and head camera over kickmsg.",
)


@app.default
def main(
    *,
    profile: Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")] = Path(
        "config/reachy_mini_sim"
    ),
    scene: Annotated[
        str | None, cyclopts.Parameter(help=f"One of {sorted(SCENES)}; sim.yaml's by default.")
    ] = None,
    viewer: bool = False,
) -> None:
    """Simulate the robot in real time until Ctrl+C or terminated.

    It publishes the joint states and the head camera's frames, and drives the joints to the
    positions clients send. The people around it are the scene's, playing their clips and saying
    their lines into `sim.yaml`'s sound sink. The robot's description, the people's avatars and
    their clips are downloaded on first use, the avatars and clips baked with Blender, and the
    lines synthesized with the robot's Kokoro, in each person's voice: the first start of a scene
    is slow, and the simulation only starts once all of it is ready.

    Args:
        profile: Robot profile directory, with `sim.yaml`, and `runtime.yaml` for Kokoro.
        scene: Who is around the robot and what they do, a function of `scenes.py`, instead of
            `sim.yaml`'s.
        viewer: Open Genesis' viewer, to watch the robot from outside.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    name = scene if scene is not None else config.scene
    description = _scene(name)
    logger.info("Setting the {} scene, of {} people", name, len(description.people))
    lines = asyncio.run(_bake(description, config.speech, RuntimeConfig.load(profile).tts))
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
        people = [
            cast(config.avatars, person, spoken, client)
            for person, spoken in zip(description.people, lines, strict=True)
        ]
        hdri = _hdri(config.camera_renderer, client)
    # One environment steps faster on the CPU than it takes to launch kernels on a GPU.
    gs.init(backend=backend.cpu, logging_level="warning")
    logging.getLogger("genesis").addFilter(_without_get_trimesh)
    logger.info("Building the scene from {}", mjcf)
    world = ReachyMiniScene(
        mjcf, config.table_height, people, config.camera_renderer, hdri, config.physics_dt, viewer
    )
    bus = RobotBus(config.namespace, world.camera_size)
    logger.info("Serving the robot in kickmsg namespace {}", config.namespace)
    # Stopped like Ctrl+C when terminated, as a launcher does, so its topics are removed too.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        with _room(config.speech, [line for spoken in lines for line in spoken]) as room:
            serve(world, bus, room, config)
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        bus.close()


async def _bake(
    scene: SceneDescription, speech: SpeechConfig, tts: KokoroConfig
) -> list[list[SpokenLine]]:
    """Each person's lines, in their voice, synthesized with the robot's Kokoro if not kept yet.

    Kept apart by the settings that change how they sound, so changing one synthesizes them anew.
    """
    settings = tts.model_dump_json(include={"model_path", "speed", "respellings"})
    directory = speech.directory / f"kokoro-{hashlib.sha256(settings.encode()).hexdigest()[:12]}"

    # One model per voice, loaded only if a line in it is missing.
    @functools.cache
    def synthesizer(voice: str) -> KokoroSynthesizer:
        return KokoroSynthesizer(tts.model_copy(update={"voice": voice}))

    return [
        await bake(person.lines, person.voice, synthesizer, directory) for person in scene.people
    ]


def _room(
    speech: SpeechConfig, lines: Sequence[SpokenLine]
) -> contextlib.AbstractContextManager[Room | None]:
    """Where the lines are played, if anywhere and if anyone speaks."""
    if speech.sink is None or not lines:
        return contextlib.nullcontext(None)
    return Room(lines, speech.sink, lines[0].clip.sample_rate)


def _scene(name: str) -> SceneDescription:
    try:
        return describe(name)
    except KeyError as error:
        raise SystemExit(error.args[0]) from error


def _hdri(renderer: CameraRenderer, client: httpx.Client) -> Path | None:
    match renderer:
        case NyxRenderer():
            return install_hdri(renderer.environment, client)
        case Rasterizer():
            return None
        case _:
            assert_never(renderer)


def _without_get_trimesh(record: logging.LogRecord) -> bool:
    """Leaves out Genesis' warning that `get_trimesh` is deprecated.

    The Nyx plugin calls it for every mesh it exports, a hundred and fifty times a start; the fix
    belongs in the plugin.
    """
    return "'get_trimesh' is deprecated" not in record.getMessage()
