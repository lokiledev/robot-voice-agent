import time
from collections.abc import Callable
from typing import Final, assert_never

from loguru import logger
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import SimConfig
from rva_robot_sim.room import Room
from rva_robot_sim.scene import ReachyMiniScene, UnknownJointError, UnknownPersonError
from rva_robot_sim.stage import SimStage
from rva_robot_sim.steps import RobotVoice, Stage
from rva_schemas.record import RobotLine, RobotReplyEnded, RobotSentence

_MAX_LAG_S: Final = 0.1
"""How far behind real time the simulation may fall before it gives up catching up."""


def serve(
    scene: ReachyMiniScene,
    bus: RobotBus,
    room: Room,
    stage: SimStage,
    script: Callable[[Stage, RobotVoice], Behaviour],
    config: SimConfig,
) -> None:
    """Simulate in real time, publishing joint states and frames, until interrupted.

    Each step first applies the commands and moves the people as received since the last one,
    then hears what the agent says the robot says, and ticks the scenario's tree, which starts
    the people's lines and clips. The people idle until a client starts the scenario: `lune`,
    once all is up, or `npc start`. Without an agent the robot says nothing, and the steps that
    wait for it time out. States and
    frames are due on simulated time, so they keep their rates on average whatever the physics
    step. A synchronous loop: the process does nothing else, and Genesis steps on this thread; the
    room plays on its audio thread.

    Whatever it publishes is stamped with the simulation's clock: the moment on the monotonic
    clock its step is due, which real time then catches up with. So a state, a frame or a
    direction carries when it happened in the simulation, without the loop's jitter, and
    clients that run on it, such as the head controller, keep pace with it; while it keeps up
    with real time, it is real time, as a robot's clock is.

    Args:
        scene: The robot and the people.
        bus: The robot's topics.
        room: Where the people's lines are played.
        stage: The people, for the scenario's steps.
        script: The scenario's tree, built when it starts.
        config: The rates.
    """
    state_due = _Due(1 / config.state_rate)
    frame_due = _Due(1 / config.camera_fps)
    doa_due = _Due(1 / config.doa_hz)
    start = time.monotonic()
    step = 0
    logger.info(
        "Simulating: joint states at {:g} Hz, frames at {:g} Hz, directions of arrival at {:g} Hz",
        config.state_rate,
        config.camera_fps,
        config.doa_hz,
    )
    logger.info("The people idle until `lune`, once all is up, or `npc start` starts the scenario")
    voice = RobotVoice()
    # The scenario's tree once started; ticked while it runs.
    tree: Behaviour | None = None
    while True:
        for request in bus.starts():
            if tree is None:
                logger.info("{} starts the scenario", request.by)
                tree = script(stage, voice)
                room.begin()
            else:
                logger.warning("{} starts the scenario, begun already", request.by)
        for command in bus.commands():
            try:
                scene.command(command)
            except UnknownJointError:
                logger.exception("Ignored a joint command")
        for pose in bus.poses():
            try:
                scene.place(pose)
            except UnknownPersonError:
                logger.exception("Ignored a person's pose")
        scene.step()
        step += 1
        simulated = step * config.physics_dt
        stage.time = simulated
        for record in bus.robot_speech():
            # In the scenario's log, to read its waits against what they waited for.
            match record:
                case RobotSentence(text=text):
                    logger.info("{:.1f} s: the robot says {!r}", simulated, text)
                case RobotLine(interrupted=interrupted):
                    logger.info(
                        "{:.1f} s: the robot {}",
                        simulated,
                        "is cut off" if interrupted else "stops",
                    )
                case RobotReplyEnded():
                    logger.info("{:.1f} s: the robot has said all it had to", simulated)
                case _:
                    assert_never(record)
            voice.hear(record, simulated)
        if tree is not None and tree.status in (Status.INVALID, Status.RUNNING):
            tree.tick_once()
            match tree.status:
                case Status.SUCCESS:
                    logger.info("The scenario ended at {:.1f} s", simulated)
                case Status.FAILURE:
                    tip = tree.tip()
                    logger.error(
                        "The scenario failed at {:.1f} s, at {!r}",
                        simulated,
                        None if tip is None else tip.name,
                    )
                case Status.RUNNING | Status.INVALID:
                    pass
        due = start + simulated
        if state_due.passed(simulated):
            bus.publish_state(scene.joint_state(due))
        if frame_due.passed(simulated):
            # The people move at the frames' rate: only the camera sees them.
            scene.animate(simulated)
            bus.publish_frame(scene.render(due))
        if doa_due.passed(simulated):
            bus.publish_sound_direction(scene.sound_direction(due))
        now = time.monotonic()
        if now - due > _MAX_LAG_S:
            logger.warning("The simulation fell {:.2f} s behind real time", now - due)
            # Carry on from now rather than rush through the backlog.
            start += now - due
        else:
            time.sleep(max(due - time.monotonic(), 0.0))


class _Due:
    """An event due every `period` seconds of simulated time."""

    def __init__(self, period: float) -> None:
        self._period: Final = period
        self._next = period

    def passed(self, simulated: float) -> bool:
        """Whether the event is due at `simulated` seconds; if so, the next one is scheduled."""
        if simulated < self._next:
            return False
        self._next += self._period
        return True
