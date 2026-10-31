import asyncio
import random
from typing import Final, Literal, Protocol, Self

from loguru import logger
from pydantic import Field, model_validator

from rva_pydantic import RVABaseModel
from rva_vla_schemas.task import FinalTaskStatus, Task, TaskFailed, TaskSucceeded

_FAILURE_REASONS: Final = (
    "the object is out of reach",
    "the grasp slipped",
    "the target could not be found",
    "the path is blocked",
    "the arm hit a joint limit",
)


class OutcomePolicy(Protocol):
    async def resolve(self, task: Task) -> FinalTaskStatus:
        """Wait until the task is over and return how it ended. Cancelled if the task is."""
        ...


class RandomPolicyConfig(RVABaseModel):
    kind: Literal["random"] = "random"
    min_duration: float = Field(ge=0)
    """Shortest task, in seconds."""
    max_duration: float = Field(ge=0)
    """Longest task, in seconds."""
    timeout: float = Field(gt=0)
    """The VLA gives up on a task still running after this many seconds."""
    failure_rate: float = Field(ge=0, le=1)
    """Share of the tasks ending in time that fail."""
    seed: int | None
    """Seed of the random generator, for reproducible runs."""

    @model_validator(mode="after")
    def _check_durations(self) -> Self:
        if self.min_duration > self.max_duration:
            raise ValueError(
                f"min_duration {self.min_duration} is above max_duration {self.max_duration}"
            )
        return self


class RandomPolicy:
    """Ends each task after a random duration: timed out past the time limit, else failing some."""

    def __init__(self, config: RandomPolicyConfig) -> None:
        self._config: Final = config
        self._rng: Final = random.Random(config.seed)

    async def resolve(self, task: Task) -> FinalTaskStatus:
        duration = self._rng.uniform(self._config.min_duration, self._config.max_duration)
        if duration > self._config.timeout:
            await asyncio.sleep(self._config.timeout)
            return _timed_out(self._config.timeout)
        await asyncio.sleep(duration)
        if self._rng.random() < self._config.failure_rate:
            return TaskFailed(cause="error", reason=self._rng.choice(_FAILURE_REASONS))
        return TaskSucceeded()


class TerminalPolicy:
    """Ends each task from a command typed by a person, or else by itself.

    Commands: `s` succeeds the task, `f [reason]` fails it, `t` times it out. A command typed while
    no task is running ends the next one. Without a command, `fallback` ends the task, if any.
    """

    def __init__(self, commands: asyncio.Queue[str], fallback: OutcomePolicy | None) -> None:
        self._commands: Final = commands
        self._fallback: Final = fallback

    async def resolve(self, task: Task) -> FinalTaskStatus:
        logger.info(
            "Task {!r}: type `s` to succeed, `f [reason]` to fail or `t` to time out",
            task.instruction,
        )
        ways = [asyncio.create_task(self._typed())]
        if self._fallback is not None:
            ways.append(asyncio.create_task(self._fallback.resolve(task)))
        try:
            done, _ = await asyncio.wait(ways, return_when=asyncio.FIRST_COMPLETED)
            return done.pop().result()
        finally:
            for way in ways:
                way.cancel()

    async def _typed(self) -> FinalTaskStatus:
        while True:
            command, _, reason = (await self._commands.get()).strip().partition(" ")
            match command:
                case "s":
                    return TaskSucceeded()
                case "f":
                    return TaskFailed(cause="error", reason=reason.strip() or "no reason given")
                case "t":
                    return TaskFailed(cause="timeout", reason="it was taking too long")
                case _:
                    logger.warning("Unknown command {!r}: type `s`, `f [reason]` or `t`", command)


def _timed_out(after: float) -> TaskFailed:
    return TaskFailed(cause="timeout", reason=f"it was still not done after {after:.0f} seconds")
