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
import rerun as rr
from loguru import logger

from rva_app.config import HeadControllerConfig
from rva_app.head_view import HeadView, read_meshes
from rva_app.ready import notify_ready
from rva_robot_client.client import RobotClient, RobotClientConfig, open_robot
from rva_robot_client.head import gaze_targets, head_targets
from rva_robot_kinematics.controller import HeadController
from rva_robot_kinematics.gaze import Gaze, HeadPoseUnknownError, HeadTrail
from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import (
    HeadKinematics,
    HeadPoseNotFoundError,
    UnreachableHeadPoseError,
)
from rva_robot_kinematics.motion import HeadMotion
from rva_robot_schemas.head import GazeTarget, HeadPose
from rva_robot_schemas.joints import JointState
from rva_robot_sim.description import install_reachy_mini

_TRAIL_S: Final = 2.0
"""How long the head's past poses are kept: longer than the camera's pipeline takes to place a
point."""
_MAX_GAP_S: Final = 0.05
"""How far from a gaze target's time the head's nearest known pose may be: five joint states at
100 Hz."""


async def control_head(
    config: HeadControllerConfig, robot_config: RobotClientConfig, viewer: bool
) -> None:
    """Move the head towards each pose received, and its camera towards each gaze target.

    The poses and targets only set where the head heads for: a loop moves it there, smoothly and
    within the limits of `config.motion`, whatever they jump. A gaze target is placed from where
    the head was when the camera saw it, so that the head's own motion since does not move it.
    Once gaze targets stop coming, the head drifts back to looking ahead. Poses the head cannot
    reach stop it where it last could go.

    Args:
        config: The head controller's.
        robot_config: Where the robot is.
        viewer: Show the robot, as the joints put it, and the targets in a rerun viewer.
    """
    mjcf = await asyncio.to_thread(_mjcf, config)
    geometry = read_head_geometry(mjcf)
    kinematics = HeadKinematics(geometry, math.radians(config.max_head_yaw_deg))
    gaze = Gaze(geometry.camera, config.gaze_position)
    view = await asyncio.to_thread(_view, mjcf) if viewer else None
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
        goal = _Goal(start, ahead, config.recentre_after_s)
        motion = HeadMotion(start, config.motion)
        trail = HeadTrail(_TRAIL_S, _MAX_GAP_S)
        logger.info("Head controller ready in kickmsg namespace {}", robot_config.namespace)
        notify_ready()
        async with asyncio.TaskGroup() as group:
            group.create_task(_follow(states, kinematics, trail, view))
            group.create_task(_look(gazes, gaze, trail, goal, view))
            group.create_task(_aim(targets, goal, view))
            await _drive(goal, motion, controller, robot, config)


class _Goal:
    """Where the head heads for.

    The last pose or gaze target sent, or ahead once gaze targets have stopped coming a while.
    """

    def __init__(self, start: HeadPose, ahead: HeadPose, recentre_after_s: float) -> None:
        self._pose = start
        self._ahead: Final = ahead
        self._recentre_after_s: Final = recentre_after_s
        self._gazed_at: float | None = None
        """When the last gaze target came, on the monotonic clock; `None` after a pose."""

    def send(self, pose: HeadPose) -> None:
        """A pose to hold until another comes, e.g. from `lune robot`."""
        self._pose = pose
        self._gazed_at = None

    def gaze(self, pose: HeadPose, now: float) -> None:
        """Where a gaze target puts the head; gaze targets keep coming while there is one."""
        self._pose = pose
        self._gazed_at = now

    def at(self, now: float) -> tuple[HeadPose, bool]:
        """The pose to head for at `now`, and whether it is ahead because the gaze stopped."""
        if self._gazed_at is not None and now - self._gazed_at > self._recentre_after_s:
            return self._ahead, True
        return self._pose, False


async def _drive(
    goal: _Goal,
    motion: HeadMotion,
    controller: HeadController,
    robot: RobotClient,
    config: HeadControllerConfig,
) -> None:
    """Move the head towards the goal at the control rate, until cancelled."""
    period = 1 / config.control_hz
    last = time.monotonic()
    reached = motion.pose
    reachable = recentring = False
    while True:
        await asyncio.sleep(period)
        now = time.monotonic()
        target, back = goal.at(now)
        if back != recentring:
            logger.info("{}", "No gaze target: looking ahead again" if back else "Gazing again")
            recentring = back
        pose = motion.step(target, now - last, config.recentre_speed_deg_s if back else None)
        last = now
        try:
            command = controller.aim(pose)
        except UnreachableHeadPoseError as error:
            if reachable:
                logger.warning("Out of reach, stopped: {} ({})", _describe(pose), error)
            reachable = False
            motion.hold(reached)
            continue
        reachable = True
        reached = pose
        robot.command(command)


async def _aim(targets: AsyncIterator[HeadPose], goal: _Goal, view: HeadView | None) -> None:
    async for target in targets:
        if view is not None:
            view.show_target(time.monotonic(), target)
        goal.send(target)
        logger.info("Head to {}", _describe(target))


async def _look(
    targets: AsyncIterator[GazeTarget],
    gaze: Gaze,
    trail: HeadTrail,
    goal: _Goal,
    view: HeadView | None,
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
        if view is not None:
            view.show_gaze(time.monotonic(), point, pose)
        goal.gaze(pose, time.monotonic())


async def _follow(
    states: AsyncIterator[JointState],
    kinematics: HeadKinematics,
    trail: HeadTrail,
    view: HeadView | None,
) -> None:
    """Note where each joint state puts the head, by forward kinematics, and show it."""
    estimate: HeadPose | None = None
    async for state in states:
        positions = {joint.name: joint.position for joint in state.joints}
        try:
            # From the last estimate, a few hundred microseconds away at 100 Hz.
            estimate = kinematics.forward(positions, estimate)
        except HeadPoseNotFoundError:
            logger.warning("No head pose fits the joints; starting over from the rest pose")
            estimate = None
            continue
        trail.add(state.timestamp, estimate)
        if view is not None:
            view.show_estimate(state.timestamp, positions["yaw_body"], estimate)


def _mjcf(config: HeadControllerConfig) -> Path:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        return install_reachy_mini(config.description, client)


def _view(mjcf: Path) -> HeadView:
    rr.init("lune head", spawn=True)
    return HeadView(read_meshes(mjcf))


def _position(state: JointState, joint: str) -> float:
    return next(reading.position for reading in state.joints if reading.name == joint)


def _describe(pose: HeadPose) -> str:
    return (
        f"({pose.x * 1000:.0f}, {pose.y * 1000:.0f}, {pose.z * 1000:.0f}) mm,"
        f" roll {math.degrees(pose.roll):.0f}, pitch {math.degrees(pose.pitch):.0f},"
        f" yaw {math.degrees(pose.yaw):.0f} deg"
    )


def _describe_point(point: npt.NDArray[np.float64]) -> str:
    x, y, z = point * 1000
    return f"({x:.0f}, {y:.0f}, {z:.0f}) mm"
