import asyncio
import os
import re
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from loguru import logger

from rva_app.servers import Server, is_ready
from rva_pydantic import RVABaseModel

_POLL_S: Final = 0.5
_STOP_TIMEOUT_S: Final = 10.0


class Service(RVABaseModel):
    """A server that `lune` starts, unless it is running already."""

    server: Server
    """How to tell it is ready."""
    command: Callable[[], Sequence[str]]
    """Called, in a thread, only when it has to be started: it may download or look up files."""
    environment: dict[str, str]
    """Variables added to this process's environment."""
    directory: Path | None
    """Where it runs, for a server that writes files to its working directory; `None` for this
    process's."""
    startup_s: float
    """How long it may take to be ready once started."""


class ServiceFailedError(Exception):
    def __init__(self, name: str, reason: str, log: Path) -> None:
        super().__init__(f"{name} {reason}; its output is in {log}")


@asynccontextmanager
async def services_running(services: Sequence[Service], logs: Path) -> AsyncGenerator[None]:
    """Start each service that is not ready yet, wait until it is, and stop it on exit.

    Services that were already running are left alone. Each one started writes its output to
    `<logs>/<name>.log`.

    Raises:
        ServiceFailedError: If a service exits, or is not ready in time.
    """
    await asyncio.to_thread(logs.mkdir, parents=True, exist_ok=True)
    started: list[asyncio.subprocess.Process] = []
    try:
        for service in services:
            if await is_ready(service.server):
                logger.info("{} is already running", service.server.name)
                continue
            log = logs / f"{_file_name(service.server.name)}.log"
            process = await _start(service, log)
            started.append(process)
            await _wait_until_ready(service, process, log)
        yield
    finally:
        await asyncio.gather(*(_stop(process) for process in started))


async def _start(service: Service, log: Path) -> asyncio.subprocess.Process:
    command = await asyncio.to_thread(service.command)
    logger.info("Starting {}; its output goes to {}", service.server.name, log)
    with log.open("wb") as output:
        return await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **service.environment},
            cwd=service.directory,
            # In a group of its own, out of reach of the terminal's Ctrl+C: stopped on exit,
            # after what talks to it, which would fail on a server gone first.
            process_group=0,
        )


async def _wait_until_ready(
    service: Service, process: asyncio.subprocess.Process, log: Path
) -> None:
    started = time.monotonic()
    while not await is_ready(service.server):
        if process.returncode is not None:
            raise ServiceFailedError(
                service.server.name, f"exited with code {process.returncode}", log
            )
        if time.monotonic() - started > service.startup_s:
            raise ServiceFailedError(
                service.server.name, f"was not ready after {service.startup_s:.0f} s", log
            )
        await asyncio.sleep(_POLL_S)
    logger.info("{} is ready", service.server.name)


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), _STOP_TIMEOUT_S)
    except TimeoutError:
        process.kill()
        await process.wait()


def _file_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
