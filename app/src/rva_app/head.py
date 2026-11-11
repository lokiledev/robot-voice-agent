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
    """Drive the head to each pose received, and its camera to each gaze target, until cancelled.

    Poses the head cannot reach are logged and left out; the head stays where it was. A gaze target
    is placed from where the head was when the camera saw it, so that the head's own motion since
    does not move it.

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
        trail = HeadTrail(_TRAIL_S, _MAX_GAP_S)
        logger.info("Head controller ready in kickmsg namespace {}", robot_config.namespace)
        async with asyncio.TaskGroup() as group:
            group.create_task(_follow(states, kinematics, trail, view))
            group.create_task(_look(gazes, gaze, trail, controller, robot, view))
            await _aim(targets, controller, robot, view)


async def _aim(
    targets: AsyncIterator[HeadPose],
    controller: HeadController,
    robot: RobotClient,
    view: HeadView | None,
) -> None:
    async for target in targets:
        if view is not None:
            view.show_target(time.monotonic(), target)
        try:
            command = controller.aim(target)
        except UnreachableHeadPoseError as error:
            logger.warning("Out of reach, left out: {} ({})", _describe(target), error)
            continue
        robot.command(command)
        logger.info("Head to {}", _describe(target))


async def _look(
    targets: AsyncIterator[GazeTarget],
    gaze: Gaze,
    trail: HeadTrail,
    controller: HeadController,
    robot: RobotClient,
    view: HeadView | None,
) -> None:
    """Point the camera at each target, logging only when targets go out of reach and back.

    The targets come with every frame the attention has one in: logging each would flood.
    """
    reachable = True
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
        try:
            command = controller.aim(pose)
        except UnreachableHeadPoseError as error:
            if reachable:
                logger.warning(
                    "Gaze targets out of reach, left out: {} needs {} ({})",
                    _describe_point(point),
                    _describe(pose),
                    error,
                )
            reachable = False
            continue
        if not reachable:
            logger.info("Gaze targets back in reach: {}", _describe_point(point))
        reachable = True
        robot.command(command)


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
