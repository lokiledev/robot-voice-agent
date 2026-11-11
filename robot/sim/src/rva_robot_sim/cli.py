import signal
from pathlib import Path
from typing import Annotated, assert_never

import cyclopts
import genesis as gs
import httpx
from genesis.constants import backend
from loguru import logger

from rva_pydantic import load_yaml
from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import CameraRenderer, NyxRenderer, Rasterizer, SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.hdri import install_hdri
from rva_robot_sim.people import install_avatar
from rva_robot_sim.scene import ReachyMiniScene
from rva_robot_sim.serve import serve

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
    viewer: bool = False,
) -> None:
    """Simulate the robot in real time until Ctrl+C or terminated.

    It publishes the joint states and the head camera's frames, and drives the joints to the
    positions clients send. The robot's description and the people's avatars are downloaded on
    first use, and the avatars baked with Blender.

    Args:
        profile: Robot profile directory, with `sim.yaml`.
        viewer: Open Genesis' viewer, to watch the robot from outside.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
        people = [
            (install_avatar(config.avatars, person.avatar, client), person)
            for person in config.people
        ]
        hdri = _hdri(config.camera_renderer, client)
    # One environment steps faster on the CPU than it takes to launch kernels on a GPU.
    gs.init(backend=backend.cpu, logging_level="warning")
    logger.info("Building the scene from {}", mjcf)
    scene = ReachyMiniScene(
        mjcf, config.table_height, people, config.camera_renderer, hdri, config.physics_dt, viewer
    )
    bus = RobotBus(config.namespace, scene.camera_size)
    logger.info("Serving the robot in kickmsg namespace {}", config.namespace)
    # Stopped like Ctrl+C when terminated, as a launcher does, so its topics are removed too.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        serve(scene, bus, config)
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        bus.close()


def _hdri(renderer: CameraRenderer, client: httpx.Client) -> Path | None:
    match renderer:
        case NyxRenderer():
            return install_hdri(renderer.environment, client)
        case Rasterizer():
            return None
        case _:
            assert_never(renderer)
