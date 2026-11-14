"""`lune-sim`: the simulated Reachy Mini, with its people and what they say."""

import asyncio
import functools
import hashlib
import logging
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, assert_never

import cyclopts
import genesis as gs
import httpx
from genesis.constants import backend
from loguru import logger

from rva_app.config import RuntimeConfig
from rva_app.ready import notify_ready
from rva_pydantic import load_yaml
from rva_robot_sim.actor import cast
from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.hdri import install_hdri
from rva_robot_sim.room import Room
from rva_robot_sim.scenarios import SCENARIOS, Live, Say, Scenario, Script, Scripted, describe
from rva_robot_sim.scene import ReachyMiniScene
from rva_robot_sim.serve import serve
from rva_robot_sim.speech import LiveMouth, ScriptedMouth, SpeechConfig, SpokenLine, bake
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
    scenario: Annotated[
        str | None,
        cyclopts.Parameter(help=f"One of {sorted(SCENARIOS)}; sim.yaml's by default."),
    ] = None,
    viewer: bool = False,
) -> None:
    """Simulate the robot in real time until Ctrl+C or terminated.

    It publishes the joint states and the head camera's frames, and drives the joints to the
    positions clients send. The people around it are the scenario's scene's, playing their
    scripts. Scripted people say their lines into the room, the sound sink the agent listens to,
    and out of the loudspeaker; a live one speaks with the microphone's voice, relayed into the
    room. The robot's description, the people's avatars and their clips are downloaded on first
    use, the avatars and clips baked with Blender, and the lines synthesized with the robot's
    Kokoro, in each person's voice: the first start of a scenario is slow, and the simulation
    only starts once all of it is ready.

    Args:
        profile: Robot profile directory, with `sim.yaml`, and `runtime.yaml` for Kokoro.
        scenario: What happens around the robot, a function of `scenarios.py`, instead of
            `sim.yaml`'s.
        viewer: Open Genesis' viewer, to watch the robot from outside.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    name = scenario if scenario is not None else config.scenario
    played = _scenario(name)
    people = played.scene.people
    logger.info("Playing the {} scenario, with {}", name, [person.name for person in people])
    lines = asyncio.run(_bake(played, config.speech, RuntimeConfig.load(profile).tts))
    voice = LiveMouth(config.speech.live_mouth)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
        try:
            actors = {
                person.name: cast(
                    config.avatars,
                    person,
                    played.scripts[person.name],
                    lines[person.name],
                    _mouth(played.scripts[person.name], lines[person.name], voice),
                    client,
                )
                for person in people
            }
        except ValueError as error:
            raise SystemExit(f"The {name} scenario: {error}") from error
        hdri = _hdri(config.camera_renderer, client)
    # One environment steps faster on the CPU than it takes to launch kernels on a GPU.
    gs.init(backend=backend.cpu, logging_level="warning")
    logging.getLogger("genesis").addFilter(_without_get_trimesh)
    logger.info("Building the scene from {}", mjcf)
    world = ReachyMiniScene(
        mjcf,
        played.scene.table_height,
        actors,
        config.camera_renderer,
        hdri,
        config.physics_dt,
        viewer,
    )
    bus = RobotBus(config.namespace, world.camera_size)
    logger.info("Serving the robot in kickmsg namespace {}", config.namespace)
    # Stopped like Ctrl+C when terminated, as a launcher does, so its topics are removed too.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        anyone_live = any(isinstance(script, Live) for script in played.scripts.values())
        scripted = [line for spoken in lines.values() for line in spoken]
        with Room(scripted, voice if anyone_live else None, config.speech) as room:
            notify_ready()
            serve(world, bus, room, config)
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        bus.close()


async def _bake(
    scenario: Scenario, speech: SpeechConfig, tts: KokoroConfig
) -> dict[str, list[SpokenLine]]:
    """Each person's lines, by name, in their voice, synthesized by the robot's Kokoro if new.

    Kept apart by the settings that change how they sound, so changing one synthesizes them anew.
    """
    settings = tts.model_dump_json(include={"model_path", "speed", "respellings"})
    directory = speech.directory / f"kokoro-{hashlib.sha256(settings.encode()).hexdigest()[:12]}"

    # One model per voice, loaded only if a line in it is missing.
    @functools.cache
    def synthesizer(voice: str) -> KokoroSynthesizer:
        return KokoroSynthesizer(tts.model_copy(update={"voice": voice}))

    spoken: dict[str, list[SpokenLine]] = {}
    for person in scenario.scene.people:
        script = scenario.scripts[person.name]
        match script:
            case Scripted(steps=steps):
                says = [step for step in steps if isinstance(step, Say)]
                spoken[person.name] = await bake(says, person.voice, synthesizer, directory)
            case Live():
                spoken[person.name] = []
            case _:
                assert_never(script)
    return spoken


def _mouth(script: Script, lines: list[SpokenLine], voice: LiveMouth) -> Callable[[float], float]:
    match script:
        case Scripted():
            return ScriptedMouth(lines).opening
        case Live():
            return voice.opening
        case _:
            assert_never(script)


def _scenario(name: str) -> Scenario:
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
