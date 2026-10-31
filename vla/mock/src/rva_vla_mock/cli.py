import asyncio
import sys

import cyclopts
import uvicorn
from loguru import logger

from rva_vla_mock.body import BodyMock
from rva_vla_mock.policies import OutcomePolicy, RandomPolicy, RandomPolicyConfig, TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_mock.server import build_app

app = cyclopts.App(
    name="rva-vla-mock", help="Serve a mock of the robot: its VLA task API and its body API."
)
# Walking and animations, with durations of their own, in seconds.
_NAVIGATION = RandomPolicyConfig(
    min_duration=5, max_duration=30, timeout=60, failure_rate=0.1, seed=None
)
_ANIMATION = RandomPolicyConfig(
    min_duration=2, max_duration=8, timeout=30, failure_rate=0.05, seed=None
)


@app.default
def main(
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    interactive: bool = False,
    min_duration: float = 5.0,
    max_duration: float = 120.0,
    timeout: float = 90.0,
    failure_rate: float = 0.2,
    seed: int | None = None,
) -> None:
    """Serve the mock robot: VLA tasks, and body motions, gestures and gaze.

    Each VLA task lasts a random time; past `timeout` it fails as timed out, otherwise some fail.
    Walking and animations behave the same with their own durations. Every call is logged.

    Args:
        host: Interface to listen on.
        port: Port to listen on.
        interactive: Also let each task be ended from the terminal before its random end.
        min_duration: Shortest random task, in seconds.
        max_duration: Longest random task, in seconds.
        timeout: The VLA gives up on a task still running after this many seconds.
        failure_rate: Share of the tasks ending in time that fail.
        seed: Seed for reproducible random outcomes.
    """
    config = RandomPolicyConfig(
        min_duration=min_duration,
        max_duration=max_duration,
        timeout=timeout,
        failure_rate=failure_rate,
        seed=seed,
    )
    asyncio.run(_serve(host, port, config, interactive))


async def _serve(host: str, port: int, config: RandomPolicyConfig, interactive: bool) -> None:
    seed = config.seed
    policies: list[OutcomePolicy] = [
        RandomPolicy(config),
        RandomPolicy(_with_seed(_NAVIGATION, None if seed is None else seed + 1)),
        RandomPolicy(_with_seed(_ANIMATION, None if seed is None else seed + 2)),
    ]
    if interactive:
        # One body action runs at a time, so whichever is running takes the typed command.
        commands = asyncio.Queue[str]()
        _forward_stdin(commands)
        policies = [TerminalPolicy(commands, fallback=policy) for policy in policies]
    vla, navigation, animation = policies
    runner = TaskRunner(vla)
    app = build_app(runner, BodyMock(runner, navigation, animation))
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port))
    logger.info("Mock robot on http://{}:{}", host, port)
    await server.serve()


def _with_seed(config: RandomPolicyConfig, seed: int | None) -> RandomPolicyConfig:
    return config.model_copy(update={"seed": seed})


def _forward_stdin(lines: asyncio.Queue[str]) -> None:
    # A reader callback rather than a thread blocked in readline, which would keep the process
    # alive after Ctrl+C until the next Enter.
    loop = asyncio.get_running_loop()

    def on_readable() -> None:
        if line := sys.stdin.readline():
            lines.put_nowait(line)
        else:
            loop.remove_reader(sys.stdin)

    loop.add_reader(sys.stdin, on_readable)


if __name__ == "__main__":
    app()
