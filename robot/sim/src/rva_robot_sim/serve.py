import time
from typing import Final

from loguru import logger

from rva_robot_sim.bus import RobotBus
from rva_robot_sim.config import SimConfig
from rva_robot_sim.room import Room
from rva_robot_sim.scene import ReachyMiniScene, UnknownJointError, UnknownPersonError

_MAX_LAG_S: Final = 0.1
"""How far behind real time the simulation may fall before it gives up catching up."""


def serve(scene: ReachyMiniScene, bus: RobotBus, room: Room, config: SimConfig) -> None:
    """Simulate in real time, publishing joint states and frames, until interrupted.

    Each step first applies the commands and moves the people as received since the last one, and
    starts the people's lines due by then. The people idle until a client starts the scenario,
    from when their steps and lines count: `lune`, once all is up, or `npc start`. States and
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
    begun: float | None = None
    while True:
        for request in bus.starts():
            if begun is None:
                begun = step * config.physics_dt
                logger.info("{} starts the scenario", request.by)
                scene.begin(begun)
                room.begin()
            else:
                logger.warning(
                    "{} starts the scenario, begun already at {:.1f} s", request.by, begun
                )
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
        if begun is not None:
            room.speak(simulated - begun)
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
