import signal
from pathlib import Path
from typing import Annotated

import cyclopts
import genesis as gs
import httpx
from genesis.constants import backend
from loguru import logger

from rva_pydantic import load_yaml
from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import SimConfig
from rva_robot_sim.description import install_reachy_mini
from rva_robot_sim.scene import ReachyMiniScene
from rva_robot_sim.serve import serve

app = cyclopts.App(
    name="lune-sim",
    help="Simulate a Reachy Mini on a table, serving its joints and head camera over kickmsg.",
)


@app.default
def main(
    *,
    profile: Annotated[Path, cyclopts.Parameter(help="Robot profile directory.")] = Path(
        "config/default"
    ),
    viewer: bool = False,
) -> None:
    """Simulate the robot in real time until Ctrl+C or terminated.

    It publishes the joint states and the head camera's frames, and drives the joints to the
    positions clients send. The robot's description is downloaded on first use.

    Args:
        profile: Robot profile directory, with `sim.yaml`.
        viewer: Open Genesis' viewer, to watch the robot from outside.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        mjcf = install_reachy_mini(config.description, client)
    # One environment steps faster on the CPU than it takes to launch kernels on a GPU.
    gs.init(backend=backend.cpu, logging_level="warning")
    logger.info("Building the scene from {}", mjcf)
    scene = ReachyMiniScene(mjcf, config.table_height, config.physics_dt, viewer)
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
