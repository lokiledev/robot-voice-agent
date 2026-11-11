import time
from typing import Final

from loguru import logger

from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import SimConfig
from rva_robot_sim.scene import ReachyMiniScene, UnknownJointError, UnknownPersonError

_MAX_LAG_S: Final = 0.1
"""How far behind real time the simulation may fall before it gives up catching up."""


def serve(scene: ReachyMiniScene, bus: RobotBus, config: SimConfig) -> None:
    """Simulate in real time, publishing joint states and frames, until interrupted.

    Each step first applies the commands and moves the people as received since the last one. States and frames are due
    on simulated time, so they keep their rates on average whatever the physics step. A synchronous
    loop: the process does nothing else, and Genesis steps on this thread.
    """
    state_due = _Due(1 / config.state_rate)
    frame_due = _Due(1 / config.camera_fps)
    start = time.monotonic()
    step = 0
    logger.info(
        "Simulating: joint states at {:g} Hz, frames at {:g} Hz",
        config.state_rate,
        config.camera_fps,
    )
    while True:
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
        now = time.monotonic()
        if state_due.passed(simulated):
            bus.publish_state(scene.joint_state(now))
        if frame_due.passed(simulated):
            bus.publish_frame(scene.render(now))
        due = start + simulated
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
