import asyncio
import sys

import cyclopts
import uvicorn
from loguru import logger

from rva_vla_mock.policies import OutcomePolicy, RandomPolicy, RandomPolicyConfig, TerminalPolicy
from rva_vla_mock.runner import TaskRunner
from rva_vla_mock.server import build_app

app = cyclopts.App(name="rva-vla-mock", help="Serve a mock of the robot's VLA task API.")


@app.default
def main(
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    interactive: bool = False,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    failure_rate: float = 0.2,
    seed: int | None = None,
) -> None:
    """Serve the mock VLA.

    Args:
        host: Interface to listen on.
        port: Port to listen on.
        interactive: End each task from the terminal instead of at random.
        min_duration: Shortest random task, in seconds.
        max_duration: Longest random task, in seconds.
        failure_rate: Share of random tasks that fail.
        seed: Seed for reproducible random outcomes.
    """
    config = RandomPolicyConfig(
        min_duration=min_duration, max_duration=max_duration, failure_rate=failure_rate, seed=seed
    )
    asyncio.run(_serve(host, port, None if interactive else config))


async def _serve(host: str, port: int, random_config: RandomPolicyConfig | None) -> None:
    policy: OutcomePolicy
    if random_config is None:
        commands = asyncio.Queue[str]()
        _forward_stdin(commands)
        policy = TerminalPolicy(commands)
    else:
        policy = RandomPolicy(random_config)
    server = uvicorn.Server(uvicorn.Config(build_app(TaskRunner(policy)), host=host, port=port))
    logger.info("Mock VLA on http://{}:{}", host, port)
    await server.serve()


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
