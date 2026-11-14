"""`lune head`: the head controller, which puts the robot's head where it is told to look.

It takes head poses, and points the head camera at gaze targets.
"""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import httpx
import numpy as np
import numpy.typing as npt
from loguru import logger

from rva_app.config import HeadControllerConfig
from rva_app.head_view import HeadView, read_meshes
from rva_app.ready import notify_ready
from rva_observability.session import ObservabilityConfig, record_session
from rva_robot_client.client import RobotClient, RobotClientConfig, open_robot
from rva_robot_client.head import (
    CameraPosePublisher,
    advertise_camera_poses,
    gaze_targets,
    head_targets,
)
from rva_robot_kinematics.controller import HeadController
from rva_robot_kinematics.gaze import Gaze, HeadPoseUnknownError, HeadTrail
from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import (
    HeadKinematics,
    HeadPoseNotFoundError,
    UnreachableHeadPoseError,
)
from rva_robot_kinematics.motion import HeadMotion
from rva_robot_schemas.head import CameraPoseStamped, GazeTarget, HeadPose
from rva_robot_schemas.joints import JointState
from rva_robot_sim.description import install_reachy_mini

_TRAIL_S: Final = 2.0
"""How long the head's past poses are kept: longer than the camera's pipeline takes to place a
point."""
_MAX_GAP_S: Final = 0.05
"""How far from a gaze target's time the head's nearest known pose may be: five joint states at
100 Hz."""


async def control_head(
    config: HeadControllerConfig,
    robot_config: RobotClientConfig,
    observability: ObservabilityConfig,
    viewer: bool,
) -> None:
    """Move the head towards each pose received, and its camera towards each gaze target.

    The poses and targets only set where the head heads for. Each joint state the robot sends
    moves it there by one step, smoothly and within the limits of `config.motion`, whatever they
    jump: the controller runs at the robot's pace, on its clock, as the simulation's slows down
    or pauses too, and sends nothing once the robot stops. A gaze target is placed from where
    the head was when the camera saw it, so that the head's own motion since does not move it.
    Once gaze targets stop coming, the head drifts back to looking ahead. Poses the head cannot
    reach stop it where it last could go.

    Args:
        config: The head controller's.
        robot_config: Where the robot is.
        observability: Where its telemetry is recorded: the robot as its joints put it, the
            poses sent and the targets, in rerun.
        viewer: Stream the telemetry to a rerun viewer too.
    """
    mjcf = await asyncio.to_thread(_mjcf, config)
    geometry = read_head_geometry(mjcf)
    kinematics = HeadKinematics(geometry, math.radians(config.max_head_yaw_deg))
    gaze = Gaze(geometry.camera, config.gaze_position)
    view = await asyncio.to_thread(_view, mjcf, observability, viewer)
    # Up before the head targets' mailbox, created last, which tells the controller runs.
    with advertise_camera_poses(robot_config.namespace) as cameras:
        async with (
            open_robot(robot_config, "head_controller_robot") as robot,
            robot.joint_states() as states,
            # Before the head targets, whose mailbox tells the controller runs: both are up then.
            gaze_targets(robot_config.namespace) as gazes,
            head_targets(robot_config.namespace) as targets,
        ):
            state = await anext(states)
            controller = HeadController(kinematics, _position(state, "yaw_body"))
            start = kinematics.forward({joint.name: joint.position for joint in state.joints}, None)
            x, y, z = config.gaze_position
            ahead = HeadPose(x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=0.0)
            goal = _Goal(ahead, config.recentre_after_s)
            trail = HeadTrail(_TRAIL_S, _MAX_GAP_S)
            driver = _Driver(
                kinematics,
                controller,
                HeadMotion(start, config.motion),
                goal,
                trail,
                gaze,
                cameras,
                robot,
                config.recentre_speed_deg_s,
                view,
            )
            logger.info("Head controller ready in kickmsg namespace {}", robot_config.namespace)
            notify_ready()
            async with asyncio.TaskGroup() as group:
                group.create_task(_look(gazes, gaze, trail, goal, view))
                group.create_task(_aim(targets, goal, view))
                await driver.run(states, state.timestamp)


class _Goal:
    """Where the head heads for.

    First ahead, at the neutral pose, as Pollen's SDK wakes the robot: from its rest pose, 28 mm
    lower, a move that lifts, pitches and turns the head at once can lead the platform where its
    servos cannot bring it back. Then the last pose or gaze target sent, or ahead again once gaze
    targets have stopped coming a while.
    """

    def __init__(self, ahead: HeadPose, recentre_after_s: float) -> None:
        self._pose = ahead
        self._ahead: Final = ahead
        self._recentre_after_s: Final = recentre_after_s
        self._awake = False
        self._gazed_at: float | None = None
        """When the frame of the last gaze target was taken, on the robot's clock; `None` after a
        pose."""

    def wake(self) -> None:
        """The head is ahead: follow what is sent from now."""
        self._awake = True

    @property
    def awake(self) -> bool:
        return self._awake

    @property
    def ahead(self) -> HeadPose:
        return self._ahead

    def send(self, pose: HeadPose) -> None:
        """A pose to hold until another comes, e.g. from `lune robot`; left out until awake."""
        if self._awake:
            self._pose = pose
            self._gazed_at = None

    def gaze(self, pose: HeadPose, seen: float) -> None:
        """Where a gaze target puts the head, seen at `seen` on the robot's clock.

        Gaze targets keep coming while there is one. Left out until awake.
        """
        if self._awake:
            self._pose = pose
            self._gazed_at = seen

    def at(self, now: float) -> tuple[HeadPose, bool]:
        """The pose to head for at `now`, on the robot's clock.

        And whether it is ahead because the gaze stopped.
        """
        if self._gazed_at is not None and now - self._gazed_at > self._recentre_after_s:
            return self._ahead, True
        return self._pose, False


class _Driver:
    """Steps the head towards its goal at each joint state, and notes where the state puts it."""

    def __init__(
        self,
        kinematics: HeadKinematics,
        controller: HeadController,
        motion: HeadMotion,
        goal: _Goal,
        trail: HeadTrail,
        gaze: Gaze,
        cameras: CameraPosePublisher,
        robot: RobotClient,
        recentre_speed_deg_s: float,
        view: HeadView,
    ) -> None:
        self._kinematics: Final = kinematics
        self._controller: Final = controller
        self._motion: Final = motion
        self._goal: Final = goal
        self._trail: Final = trail
        self._gaze: Final = gaze
        self._cameras: Final = cameras
        self._robot: Final = robot
        self._recentre_speed_deg_s: Final = recentre_speed_deg_s
        self._view: Final = view
        self._estimate: HeadPose | None = None
        self._reached = motion.pose
        self._reachable = True
        self._recentring = False

    async def run(self, states: AsyncIterator[JointState], start: float) -> None:
        """Follow the robot's states from the one at `start`, until they stop.

        Raises:
            RobotUnavailableError: Once the robot has been silent too long.
        """
        last = start
        async for state in states:
            self._note(state)
            # Over however long the robot took: a state skipped makes a longer step.
            self._step(state.timestamp, state.timestamp - last)
            last = state.timestamp

    def _note(self, state: JointState) -> None:
        """Note where the state puts the head, by forward kinematics, and show it."""
        positions = {joint.name: joint.position for joint in state.joints}
        try:
            # From the last estimate, a few hundred microseconds away at 100 Hz.
            self._estimate = self._kinematics.forward(positions, self._estimate)
        except HeadPoseNotFoundError:
            logger.warning("No head pose fits the joints; starting over from the rest pose")
            self._estimate = None
            return
        self._trail.add(state.timestamp, self._estimate)
        camera = self._gaze.camera(self._estimate)
        rows = camera[:3, :3].tolist()
        self._cameras.publish(
            CameraPoseStamped(
                timestamp=state.timestamp,
                position=(float(camera[0, 3]), float(camera[1, 3]), float(camera[2, 3])),
                rotation=(tuple(rows[0]), tuple(rows[1]), tuple(rows[2])),
            )
        )
        self._view.show_estimate(state.timestamp, positions["yaw_body"], self._estimate)

    def _step(self, now: float, dt: float) -> None:
        if not self._goal.awake and _near(self._motion.pose, self._goal.ahead):
            logger.info("Head up and ahead: following what it is sent")
            self._goal.wake()
        target, back = self._goal.at(now)
        if back != self._recentring:
            logger.info("{}", "No gaze target: looking ahead again" if back else "Gazing again")
            self._recentring = back
        pose = self._motion.step(target, dt, self._recentre_speed_deg_s if back else None)
        self._view.show_sent(now, pose, target)
        try:
            command = self._controller.aim(pose)
        except UnreachableHeadPoseError as error:
            if self._reachable:
                logger.warning("Out of reach, stopped: {} ({})", _describe(pose), error)
            self._reachable = False
            self._motion.hold(self._reached)
            return
        self._reachable = True
        self._reached = pose
        self._robot.command(command)


async def _aim(targets: AsyncIterator[HeadPose], goal: _Goal, view: HeadView) -> None:
    async for target in targets:
        view.show_target(time.monotonic(), target)
        goal.send(target)
        logger.info("Head to {}", _describe(target))


async def _look(
    targets: AsyncIterator[GazeTarget],
    gaze: Gaze,
    trail: HeadTrail,
    goal: _Goal,
    view: HeadView,
) -> None:
    """Head the camera for each target, placed from where the head was when the camera saw it."""
    async for target in targets:
        try:
            head = trail.at(target.timestamp)
        except HeadPoseUnknownError as error:
            logger.warning("Gaze target left out: {}", error)
            continue
        point = gaze.place(target, head)
        pose = gaze.aim(point)
        view.show_gaze(target.timestamp, point, pose)
        goal.gaze(pose, target.timestamp)


def _mjcf(config: HeadControllerConfig) -> Path:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        return install_reachy_mini(config.description, client)


def _view(mjcf: Path, observability: ObservabilityConfig, viewer: bool) -> HeadView:
    return HeadView(read_meshes(mjcf), record_session(observability, "head", viewer))


def _position(state: JointState, joint: str) -> float:
    return next(reading.position for reading in state.joints if reading.name == joint)


def _near(pose: HeadPose, other: HeadPose) -> bool:
    """Within a millimetre and a tenth of a degree on each axis."""
    return all(
        abs(getattr(pose, axis) - getattr(other, axis)) < 0.001 for axis in ("x", "y", "z")
    ) and all(
        abs(getattr(pose, axis) - getattr(other, axis)) < math.radians(0.1)
        for axis in ("roll", "pitch", "yaw")
    )


def _describe(pose: HeadPose) -> str:
    return (
        f"({pose.x * 1000:.0f}, {pose.y * 1000:.0f}, {pose.z * 1000:.0f}) mm,"
        f" roll {math.degrees(pose.roll):.0f}, pitch {math.degrees(pose.pitch):.0f},"
        f" yaw {math.degrees(pose.yaw):.0f} deg"
    )


def _describe_point(point: npt.NDArray[np.float64]) -> str:
    x, y, z = point * 1000
    return f"({x:.0f}, {y:.0f}, {z:.0f}) mm"
