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
    failure_rate: float = Field(ge=0, le=1)
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
    """Ends each task after a random duration, failing some of them."""

    def __init__(self, config: RandomPolicyConfig) -> None:
        self._config: Final = config
        self._rng: Final = random.Random(config.seed)

    async def resolve(self, task: Task) -> FinalTaskStatus:
        await asyncio.sleep(self._rng.uniform(self._config.min_duration, self._config.max_duration))
        if self._rng.random() < self._config.failure_rate:
            return TaskFailed(reason=self._rng.choice(_FAILURE_REASONS))
        return TaskSucceeded()


class TerminalPolicy:
    """Ends each task from a command typed by a person.

    Commands: `s` succeeds the task, `f [reason]` fails it. A command typed while no task is running
    ends the next one.
    """

    def __init__(self, commands: asyncio.Queue[str]) -> None:
        self._commands: Final = commands

    async def resolve(self, task: Task) -> FinalTaskStatus:
        logger.info("Task {!r}: type `s` to succeed or `f [reason]` to fail", task.instruction)
        while True:
            command, _, reason = (await self._commands.get()).strip().partition(" ")
            match command:
                case "s":
                    return TaskSucceeded()
                case "f":
                    return TaskFailed(reason=reason.strip() or "no reason given")
                case _:
                    logger.warning("Unknown command {!r}: type `s` or `f [reason]`", command)
