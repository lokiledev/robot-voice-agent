"""The steps of a scenario that waits on the robot, as the leaves of a py_trees behavior tree.

A scenario is a tree: `py_trees.composites.Sequence` with memory for one step after the other,
`Parallel` for steps at once, and these leaves, which start something on the stage and run until
it ends, or wait. A wait the robot does not meet fails, and so does the sequence it is in: a demo
goes on regardless with `anyway`, a test lets it fail. Ticked by whoever runs the scenario, once
per step of its clock; py_trees' own timers run on the wall clock and are not used.
"""

from abc import abstractmethod
from typing import Final, Protocol, assert_never, override

from loguru import logger
from py_trees.behaviour import Behaviour
from py_trees.common import ParallelPolicy, Status
from py_trees.composites import Parallel
from py_trees.decorators import FailureIsSuccess

from rva_robot_sim.animation_assets import ClipName
from rva_schemas.record import RobotLine, RobotReplyEnded, RobotSentence


class Stage(Protocol):
    """Where the scenario's people act: the simulation, or a test's fake."""

    def now(self) -> float:
        """The time on the stage's clock, in seconds: the simulation's."""
        ...

    def say(self, who: str, text: str, gesture: ClipName) -> float:
        """Start `who` saying `text`, gesturing as long as it lasts.

        Returns:
            How long the line lasts, in seconds.
        """
        ...

    def play(self, who: str, clips: tuple[ClipName, ...]) -> float:
        """Start `who` playing `clips` once, one after the other.

        Returns:
            How long they last together, in seconds.
        """
        ...

    def cut(self, who: str) -> None:
        """Stop what `who` says or plays, before its end: they idle from where they are."""
        ...


class Unstaged:
    """A `Stage` for a tree that is only read, to know its lines and clips before it is played."""

    def now(self) -> float:
        raise RuntimeError("Read, not played")

    def say(self, who: str, text: str, gesture: ClipName) -> float:
        raise RuntimeError("Read, not played")

    def play(self, who: str, clips: tuple[ClipName, ...]) -> float:
        raise RuntimeError("Read, not played")

    def cut(self, who: str) -> None:
        raise RuntimeError("Read, not played")


class RobotVoice:
    """When the robot speaks and when it has said all it had to, from what its agent records.

    A stretch of speech starts with its first `RobotSentence` and ends with its `RobotLine`. A
    reply is one or several stretches, with silences while the robot calls a tool or thinks of
    what else it heard; it ends with a `RobotReplyEnded`.
    """

    def __init__(self) -> None:
        self.started_at: float | None = None
        """When the last stretch started; `None` until the robot first speaks."""
        self.stopped_at: float | None = None
        """When the last stretch ended; `None` until one has."""
        self.reply_ended_at: float | None = None
        """When the last reply ended; `None` until one has."""
        self.replying = False
        """Whether the robot has started a reply it has not ended, speaking now or not."""

    @property
    def speaking(self) -> bool:
        return self.started_at is not None and (
            self.stopped_at is None or self.stopped_at < self.started_at
        )

    def hear(self, record: RobotSentence | RobotLine | RobotReplyEnded, time: float) -> None:
        """Take a record of the agent's, received at `time`, in seconds of the stage's clock."""
        match record:
            case RobotSentence():
                if not self.speaking:
                    self.started_at = time
                self.replying = True
            case RobotLine():
                self.stopped_at = time
                # Typed to, the robot answers in lines without sentences.
                self.replying = True
            case RobotReplyEnded():
                self.reply_ended_at = time
                self.replying = False
            case _:
                assert_never(record)


class _Lasting(Behaviour):
    """Starts something that lasts a known time, and runs until it ends."""

    def __init__(self, name: str, stage: Stage) -> None:
        super().__init__(name)
        self._stage: Final = stage
        self._ends = 0.0

    @override
    def initialise(self) -> None:
        self._ends = self._stage.now() + self._start()

    @override
    def update(self) -> Status:
        return Status.SUCCESS if self._stage.now() >= self._ends else Status.RUNNING

    @abstractmethod
    def _start(self) -> float:
        """Start it. Returns how long it lasts, in seconds."""


class _Acting(_Lasting):
    """Something a person does, cut if the tree stops it before its end."""

    def __init__(self, name: str, stage: Stage, who: str) -> None:
        super().__init__(name, stage)
        self.who: Final = who

    @override
    def terminate(self, new_status: Status) -> None:
        # Stopped from outside while running, e.g. by a parallel that ended without it.
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self._stage.cut(self.who)


class Say(_Acting):
    """A person says a line, gesturing as long as it lasts."""

    def __init__(self, stage: Stage, who: str, text: str, gesture: ClipName) -> None:
        super().__init__(f"{who}: {text}", stage, who)
        self.text: Final = text
        self.gesture: Final = gesture

    @override
    def _start(self) -> float:
        return self._stage.say(self.who, self.text, self.gesture)


class Play(_Acting):
    """A person plays clips once, one after the other."""

    def __init__(self, stage: Stage, who: str, clips: tuple[ClipName, ...]) -> None:
        super().__init__(f"{who} plays {', '.join(clip.name for clip in clips)}", stage, who)
        self.clips: Final = clips

    @override
    def _start(self) -> float:
        return self._stage.play(self.who, self.clips)


class Wait(_Lasting):
    """Nothing, for a time."""

    def __init__(self, stage: Stage, seconds: float) -> None:
        super().__init__(f"wait {seconds:g} s", stage)
        self._seconds: Final = seconds

    @override
    def _start(self) -> float:
        return self._seconds


class WaitForRobot(Behaviour):
    """Wait until the robot has said all it has to.

    Succeeds once a reply of the robot's ends, one under way when the wait began or begun since:
    every stretch of it, as the agent says, not a silence in it. Fails if the robot has not
    started one `within_s` after the wait began; once it has, it is waited for however long it
    goes on.
    """

    def __init__(self, stage: Stage, robot: RobotVoice, within_s: float) -> None:
        super().__init__(f"wait for the robot, {within_s:g} s for it to start")
        self._stage: Final = stage
        self._robot: Final = robot
        self._within_s: Final = within_s
        self._began = 0.0

    @override
    def initialise(self) -> None:
        self._began = self._stage.now()

    @override
    def update(self) -> Status:
        if self._robot.replying:
            return Status.RUNNING
        ended = self._robot.reply_ended_at
        if ended is not None and ended >= self._began:
            return Status.SUCCESS
        if self._stage.now() - self._began >= self._within_s:
            logger.warning("The robot said nothing for {:g} s", self._within_s)
            return Status.FAILURE
        return Status.RUNNING


class WaitUntilRobotSpoke(Behaviour):
    """Wait until the robot has been speaking for a while, to talk over it.

    Succeeds once it has spoken for `for_s` without stopping. Fails `within_s` after the wait
    began if it has not.
    """

    def __init__(self, stage: Stage, robot: RobotVoice, for_s: float, within_s: float) -> None:
        super().__init__(f"wait until the robot spoke for {for_s:g} s, {within_s:g} s at most")
        self._stage: Final = stage
        self._robot: Final = robot
        self._for_s: Final = for_s
        self._within_s: Final = within_s
        self._began = 0.0

    @override
    def initialise(self) -> None:
        self._began = self._stage.now()

    @override
    def update(self) -> Status:
        now = self._stage.now()
        started = self._robot.started_at
        if self._robot.speaking and started is not None and now - started >= self._for_s:
            return Status.SUCCESS
        if now - self._began >= self._within_s:
            logger.warning("The robot did not speak for {:g} s on end", self._for_s)
            return Status.FAILURE
        return Status.RUNNING


def anyway(step: Behaviour) -> Behaviour:
    """`step`, going on as if it had succeeded when it fails: a wait the robot does not meet."""
    return FailureIsSuccess(f"{step.name}, or go on", step)


def alongside(main: Behaviour, *background: Behaviour) -> Behaviour:
    """`main` with `background` steps started at once, which are cut when `main` ends.

    For what only fills the time: someone nodding while another speaks.
    """
    return Parallel(
        f"{main.name}, alongside",
        ParallelPolicy.SuccessOnSelected([main]),
        [main, *background],
    )
