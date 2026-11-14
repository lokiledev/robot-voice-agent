import asyncio
import contextlib
import os
import re
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from loguru import logger

from rva_app.ready import READY, READY_FD
from rva_app.servers import Server, is_ready
from rva_pydantic import RVABaseModel

_POLL_S: Final = 0.5
_STOP_TIMEOUT_S: Final = 10.0


class Service(RVABaseModel):
    """A server that `lune` starts, unless it is running already."""

    server: Server
    """How to tell it is running already, and ready once started if it does not notify."""
    notifies: bool
    """Whether it says it is ready on the launcher's pipe (`ready.notify_ready`), as the
    processes of this project do, rather than being probed until it answers."""
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
    def __init__(self, name: str, reason: str, log: Path | None) -> None:
        where = "" if log is None else f"; its output is in {log}"
        super().__init__(f"{name} {reason}{where}")


@asynccontextmanager
async def services_running(services: Sequence[Service], logs: Path) -> AsyncGenerator[None]:
    """Start each service that is not running yet, one after the other, and stop them on exit.

    Each starts once the one before is ready. Services that were already running are left alone. Each one started writes its output to
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
            process, ready = await _start(service, log)
            started.append(process)
            await _wait_until_ready(service, process, ready, log)
        yield
    finally:
        await asyncio.gather(*(_stop(process) for process in started))


async def _start(service: Service, log: Path) -> tuple[asyncio.subprocess.Process, int | None]:
    """The process started, and the read end of its ready pipe if it notifies."""
    command = await asyncio.to_thread(service.command)
    logger.info("Starting {}; its output goes to {}", service.server.name, log)
    with log.open("wb") as output, _ready_pipe(service.notifies) as (ready, environment):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **service.environment, **environment},
            cwd=service.directory,
            pass_fds=tuple(int(fd) for fd in environment.values()),
            # In a group of its own, out of reach of the terminal's Ctrl+C: stopped on exit,
            # after what talks to it, which would fail on a server gone first.
            process_group=0,
        )
    return process, ready


async def _wait_until_ready(
    service: Service, process: asyncio.subprocess.Process, ready: int | None, log: Path
) -> None:
    started = time.monotonic()
    while not await _is_ready(service, ready):
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


async def _is_ready(service: Service, ready: int | None) -> bool:
    if ready is None:
        return await is_ready(service.server)
    return _notified(ready)


async def run_last(
    name: str, command: Sequence[str], startup_s: float, when_ready: Callable[[], None]
) -> int:
    """Run `command` in the terminal, call `when_ready` once it says it is ready, and wait for it.

    It shares the terminal: its input, its output, and its Ctrl+C, after which it is given
    time to stop by itself.

    Args:
        name: For the log.
        command: A process that notifies (`ready.notify_ready`).
        startup_s: How long it may take to be ready.
        when_ready: Called once, when it is.

    Returns:
        Its exit code.

    Raises:
        ServiceFailedError: If it exits, or is not ready in time, before it is ready.
    """
    logger.info("Starting {}", name)
    with _ready_pipe(notifies=True) as (ready, environment):
        process = await asyncio.create_subprocess_exec(
            *command,
            env={**os.environ, **environment},
            pass_fds=tuple(int(fd) for fd in environment.values()),
        )
    assert ready is not None, "a pipe for a process that notifies"
    try:
        started = time.monotonic()
        while not _notified(ready):
            if process.returncode is not None:
                raise ServiceFailedError(name, f"exited with code {process.returncode}", None)
            if time.monotonic() - started > startup_s:
                raise ServiceFailedError(name, f"was not ready after {startup_s:.0f} s", None)
            await asyncio.sleep(_POLL_S)
        logger.info("{} is ready", name)
        when_ready()
        return await process.wait()
    finally:
        # Ctrl+C reached it too: it is left time to stop by itself and save its recording.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), _STOP_TIMEOUT_S)
        await _stop(process)


@contextlib.contextmanager
def _ready_pipe(notifies: bool) -> Generator[tuple[int | None, dict[str, str]]]:
    """A ready pipe for a process about to start, if it notifies.

    Gives the pipe's read end, and the variable that tells the process its write end, which is
    closed here once the process has it.
    """
    if not notifies:
        yield None, {}
        return
    read, write = os.pipe()
    os.set_blocking(read, False)
    try:
        yield read, {READY_FD: str(write)}
    finally:
        os.close(write)


def _notified(ready: int) -> bool:
    """Whether the process wrote it is ready; the pipe is closed once it has."""
    try:
        said = os.read(ready, len(READY))
    except BlockingIOError:
        return False
    # Its end closed without a word: it exited, which the caller sees.
    if said != READY:
        return False
    os.close(ready)
    return True


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
