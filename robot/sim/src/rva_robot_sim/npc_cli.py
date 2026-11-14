import asyncio
from pathlib import Path
from typing import Annotated

import cyclopts

from rva_pydantic import load_yaml
from rva_robot_sim.config import SimConfig
from rva_robot_sim.npc import (
    SimulationNotRunningError,
    SimulationStoppedError,
    open_people_mover,
    pace,
    start_scenario,
)
from rva_robot_sim.scenarios import SCENARIOS, describe

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
    scenario: Annotated[
        str | None, cyclopts.Parameter(help=f"One of {sorted(SCENARIOS)}; sim.yaml's by default.")
    ] = None,
    person: str = "ana",
    half_width: float = 0.8,
    speed: float = 0.4,
) -> None:
    """Pace one of the simulated people sideways across the robot's view, back and forth.

    To watch the head follow them, with `lune head` and `lune see`. Start the simulation first with
    `uv run lune-sim`; Ctrl+C stops, and puts them back where `sim.yaml` places them.

    Args:
        profile: Robot profile directory, with the `sim.yaml` the simulation runs.
        scenario: The scenario the simulation runs, if `--scenario` gave it another than
            `sim.yaml`'s.
        person: Which of the scene's people, by name.
        half_width: How far they go each side of where they stand, in meters.
        speed: How fast they walk, in meters per second.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    try:
        walker = describe(scenario if scenario is not None else config.scenario).scene.person(
            person
        )
    except KeyError as error:
        raise SystemExit(error.args[0]) from error
    mover = open_people_mover(config.namespace, "npc_walk")
    if mover is None:
        raise SystemExit(
            f"No simulation in kickmsg namespace {config.namespace!r}; start it with"
            f" `uv run lune-sim --profile {profile}`"
        )
    try:
        asyncio.run(pace(mover, walker, half_width, speed))
    except SimulationStoppedError as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass


@app.command
def start(*, profile: ProfileDir = Path("config/reachy_mini_sim")) -> None:
    """Start the scenario of a running `lune-sim`, whose people idle until then.

    The agent starts it once up; this is for running the simulation without the agent, e.g. with
    `lune see`. A scenario starts once: starting it again only logs a warning in the simulation.

    Args:
        profile: Robot profile directory, with the `sim.yaml` the simulation runs.
    """
    config = load_yaml(SimConfig, profile / "sim.yaml")
    try:
        start_scenario(config.namespace, "npc start")
    except SimulationNotRunningError as error:
        raise SystemExit(f"{error}; start it with `uv run lune-sim --profile {profile}`") from error
