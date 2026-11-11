import asyncio
from pathlib import Path
from typing import Annotated

import cyclopts

from rva_pydantic import load_yaml
from rva_robot_sim.config import SimConfig
from rva_robot_sim.npc import SimulationStoppedError, open_people_mover, pace

app = cyclopts.App(
    name="npc",
    help="Control the people of a running `lune-sim`, its NPCs, from the outside.",
)

type ProfileDir = Annotated[
    Path, cyclopts.Parameter(help="Robot profile directory, with the simulation's `sim.yaml`.")
]


@app.command
def walk(
    *,
    profile: ProfileDir = Path("config/reachy_mini_sim"),
    person: int = 0,
    half_width: float = 0.8,
    speed: float = 0.4,
) -> None:
    """Pace one of the simulated people sideways across the robot's view, back and forth.

    To watch the head follow them, with `lune head` and `lune see`. Start the simulation first with
    `uv run lune-sim`; Ctrl+C stops, and puts them back where `sim.yaml` places them.

    Args:
        profile: Robot profile directory, with the `sim.yaml` the simulation runs.
        person: Which of `sim.yaml`'s people, by index.
        half_width: How far they go each side of where they stand, in meters.
        speed: How fast they walk, in meters per second.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    if not 0 <= person < len(config.people):
        raise SystemExit(f"{profile / 'sim.yaml'} has {len(config.people)} people, from 0")
    mover = open_people_mover(config.namespace, "npc_walk")
    if mover is None:
        raise SystemExit(
            f"No simulation in kickmsg namespace {config.namespace!r}; start it with"
            f" `uv run lune-sim --profile {profile}`"
        )
    try:
        asyncio.run(pace(mover, person, config.people[person], half_width, speed))
    except SimulationStoppedError as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass
