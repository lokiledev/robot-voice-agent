"""`lune head`: the head controller, which puts the robot's head at the poses it is sent."""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import rerun as rr
from loguru import logger

from rva_app.config import HeadControllerConfig
from rva_app.head_view import HeadView, read_meshes
from rva_robot_client.client import RobotClient, RobotClientConfig, open_robot
from rva_robot_client.head import head_targets
from rva_robot_kinematics.controller import HeadController
from rva_robot_kinematics.geometry import read_head_geometry
from rva_robot_kinematics.head import (
    HeadKinematics,
    HeadPoseNotFoundError,
    UnreachableHeadPoseError,
)
from rva_robot_schemas.head import HeadPose
from rva_robot_schemas.joints import JointState
from rva_robot_sim.description import install_reachy_mini


async def control_head(
    config: HeadControllerConfig, robot_config: RobotClientConfig, viewer: bool
) -> None:
    """Drive the head to each pose received, until cancelled.

    Poses the head cannot reach are logged and left out; the head stays where it was.

    Args:
        config: The head controller's.
        robot_config: Where the robot is.
        viewer: Show the robot, as the joints put it, and the targets in a rerun viewer.
    """
    mjcf = await asyncio.to_thread(_mjcf, config)
    kinematics = HeadKinematics(read_head_geometry(mjcf), math.radians(config.max_head_yaw_deg))
    view = await asyncio.to_thread(_view, mjcf) if viewer else None
    async with (
        open_robot(robot_config, "head_controller_robot") as robot,
        robot.joint_states() as states,
        head_targets(robot_config.namespace) as targets,
    ):
        state = await anext(states)
        controller = HeadController(kinematics, _position(state, "yaw_body"))
        logger.info("Head controller ready in kickmsg namespace {}", robot_config.namespace)
        async with asyncio.TaskGroup() as group:
            if view is not None:
                group.create_task(_show(states, kinematics, view))
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


async def _show(
    states: AsyncIterator[JointState], kinematics: HeadKinematics, view: HeadView
) -> None:
    """Show the robot as each joint state puts it, the head where forward kinematics finds it."""
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
