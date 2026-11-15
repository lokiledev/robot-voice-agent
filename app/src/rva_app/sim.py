"""`lune-sim`: the simulated Reachy Mini, with its people and what they say."""

import asyncio
import functools
import hashlib
import logging
import signal
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
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.hdri import install_hdri
from rva_robot_sim.room import Room
from rva_robot_sim.scenarios import SCENARIOS, Scenario, describe
from rva_robot_sim.scene import ReachyMiniScene
from rva_robot_sim.serve import serve
from rva_robot_sim.speech import LiveMouth, ScriptedMouth, SpeechConfig, bake
from rva_robot_sim.stage import SimStage
from rva_robot_sim.steps import Play, Say
from rva_schemas.audio import AudioClip
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
    positions clients send. The people around it are the scenario's scene's, playing its script,
    which waits on what the agent says the robot says. They say their lines into the room, the
    sound sink the agent listens to, and out of the loudspeaker; a live one speaks with the
    microphone's voice, relayed into the room. The robot's description, the people's avatars and their clips are downloaded on first
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
    acts = played.acts()
    lines = asyncio.run(_bake(played, acts, config.speech, RuntimeConfig.load(profile).tts))
    voice = LiveMouth(config.speech.live_mouth)
    mouths = {person.name: ScriptedMouth() for person in people if person.name != played.live}
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
        cast_people = {
            person.name: cast(
                config.avatars,
                person,
                played.idle[person.name],
                _clips(acts, person.name),
                mouths[person.name].opening if person.name in mouths else voice.opening,
                client,
            )
            for person in people
        }
        hdri = _hdri(config.camera_renderer, client)
    # One environment steps faster on the CPU than it takes to launch kernels on a GPU.
    gs.init(backend=backend.cpu, logging_level="warning")
    logging.getLogger("genesis").addFilter(_without_get_trimesh)
    logger.info("Building the scene from {}", mjcf)
    world = ReachyMiniScene(
        mjcf,
        played.scene.table_height,
        cast_people,
        config.servos,
        config.passive_joints,
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
        live = voice if played.live is not None else None
        with Room(list(lines.values()), live, config.speech) as room:
            actors = {name: actor for name, (_, actor) in cast_people.items()}
            stage = SimStage(actors, mouths, lines, room)
            notify_ready()
            serve(world, bus, room, stage, played.script, config)
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        bus.close()


async def _bake(
    scenario: Scenario, acts: tuple[Say | Play, ...], speech: SpeechConfig, tts: KokoroConfig
) -> dict[tuple[str, str], AudioClip]:
    """Each line of the script, by who says it and its text, synthesized by the robot's Kokoro.

    In each person's voice, and only if new. Kept apart by the settings that change how they
    sound, so changing one synthesizes them anew.
    """
    settings = tts.model_dump_json(include={"model_path", "speed", "respellings"})
    directory = speech.directory / f"kokoro-{hashlib.sha256(settings.encode()).hexdigest()[:12]}"

    # One model per voice, loaded only if a line in it is missing.
    @functools.cache
    def synthesizer(voice: str) -> KokoroSynthesizer:
        return KokoroSynthesizer(tts.model_copy(update={"voice": voice}))

    spoken: dict[tuple[str, str], AudioClip] = {}
    for person in scenario.scene.people:
        texts = [act.text for act in acts if isinstance(act, Say) and act.who == person.name]
        baked = await bake(texts, person.voice, synthesizer, directory)
        spoken.update({(person.name, text): clip for text, clip in baked.items()})
    return spoken


def _clips(acts: tuple[Say | Play, ...], who: str) -> list[ClipName]:
    """Every clip the script has `who` play or gesture with."""
    clips: list[ClipName] = []
    for act in acts:
        if act.who != who:
            continue
        match act:
            case Say(gesture=gesture):
                clips.append(gesture)
            case Play(clips=played):
                clips.extend(played)
            case _:
                assert_never(act)
    return clips


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
