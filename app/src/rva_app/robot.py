"""`lune robot`: drive the robot's joints with sliders and watch its head camera."""

import asyncio
import math
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt

from rva_robot_client.client import RobotClient, RobotClientConfig, open_robot
from rva_robot_schemas.camera import CameraFrame
from rva_robot_schemas.joints import JointCommand, JointState, JointTarget

_TITLE: Final = "lune robot"
_IMAGE_SIZE: Final = (960, 540)
_SLIDER_WIDTH: Final = 220
_PANEL_WIDTH: Final = 420
"""Room for the sliders, their labels and the measured positions."""
_MARGIN: Final = 40
_RANGE_DEG: Final = 180.0
"""Each slider's reach either way; the robot clamps targets to the joint's own range."""
_LIFT: Final = {f"stewart_{motor}": 1.0 if motor % 2 else -1.0 for motor in range(1, 7)}
"""Odd and even motors of Reachy Mini's Stewart platform turn in opposite senses to lift the head
without tilting it."""
_LIFT_RANGE_DEG: Final = 30.0


async def drive_robot(config: RobotClientConfig) -> None:
    """Show the head camera and the joints, and send the targets set with the sliders.

    Runs until the window is closed.
    """
    async with (
        open_robot(config, "lune_robot") as robot,
        robot.joint_states() as states,
        robot.frames() as frames,
    ):
        driver = _Driver(robot, await anext(states))
        # Dear PyGui is driven from one thread, which opens, draws and closes its window.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-window") as thread:
            loop = asyncio.get_running_loop()
            window = await loop.run_in_executor(thread, _RobotWindow, driver.targets)
            try:
                async with asyncio.TaskGroup() as group:
                    following = group.create_task(driver.follow(states))
                    await driver.show(frames, window, thread)
                    following.cancel()
            finally:
                await loop.run_in_executor(thread, window.close)


class _Driver:
    def __init__(self, robot: RobotClient, state: JointState) -> None:
        self._robot: Final = robot
        self._state = state
        self.targets: Final = {joint.name: joint.position for joint in state.joints}
        """Where each joint is sent, in radians, from where it was."""

    async def follow(self, states: AsyncIterator[JointState]) -> None:
        async for state in states:
            self._state = state

    async def show(
        self, frames: AsyncIterator[CameraFrame], window: "_RobotWindow", thread: ThreadPoolExecutor
    ) -> None:
        """Show each frame and send the targets changed meanwhile, until the window is closed."""
        loop = asyncio.get_running_loop()
        async for frame in frames:
            changed = await loop.run_in_executor(thread, window.show, frame.image, self._state)
            if changed is None:
                return
            if changed:
                self.targets.update(changed)
                self._robot.command(
                    JointCommand(
                        targets=tuple(JointTarget(name=n, position=p) for n, p in changed.items())
                    )
                )


class _RobotWindow:
    """The head camera, and a slider per joint with where it is.

    Lives in the thread that creates it, which also shows the frames and closes it.
    """

    def __init__(self, targets: Mapping[str, float]) -> None:
        self._image: Final = np.zeros((_IMAGE_SIZE[1], _IMAGE_SIZE[0], 3), np.float32)
        self._changed: dict[str, float] = {}
        """Targets set since the last frame, in radians."""
        dpg.create_context()
        # Callbacks run when `show` asks, in this thread, rather than in one of Dear PyGui's.
        dpg.configure_app(manual_callback_management=True)
        with dpg.texture_registry():
            self._texture: Final = dpg.add_raw_texture(
                *_IMAGE_SIZE,
                self._image,  # ty: ignore[invalid-argument-type]  # Any buffer is read; the stubs only list lists.
                format=dpg.mvFormat_Float_rgb,
            )
        self._sliders: Final[dict[str, int | str]] = {}
        self._readings: Final[dict[str, int | str]] = {}
        with dpg.window() as root:
            with dpg.group(horizontal=True):
                dpg.add_image(self._texture)
                with dpg.group():
                    dpg.add_text("Targets, in degrees; measured on the right")
                    for name, position in targets.items():
                        with dpg.group(horizontal=True):
                            self._sliders[name] = dpg.add_slider_float(
                                label=name,
                                default_value=math.degrees(position),
                                min_value=-_RANGE_DEG,
                                max_value=_RANGE_DEG,
                                width=_SLIDER_WIDTH,
                                callback=self._on_slider,
                                user_data=name,
                            )
                            self._readings[name] = dpg.add_text()
                    dpg.add_spacer(height=10)
                    self._lift: Final = dpg.add_slider_float(
                        label="head lift",
                        min_value=-_LIFT_RANGE_DEG,
                        max_value=_LIFT_RANGE_DEG,
                        width=_SLIDER_WIDTH,
                        callback=self._on_lift,
                    )
                    dpg.add_button(label="Rest pose", callback=self._on_rest)
        dpg.create_viewport(
            title=_TITLE,
            width=_IMAGE_SIZE[0] + _PANEL_WIDTH + 2 * _MARGIN,
            height=_IMAGE_SIZE[1] + _MARGIN,
            # Drawing must not wait for the screen, or frames would queue up behind it.
            vsync=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(root, True)

    def show(self, frame: npt.NDArray[np.uint8], state: JointState) -> dict[str, float] | None:
        """Show a frame and the joints' positions.

        Returns:
            The targets set with the sliders since the last frame, in radians; `None` once the
            window is closed.
        """
        if not dpg.is_dearpygui_running():
            return None
        small = cv2.resize(frame, _IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        np.multiply(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self._image)
        dpg.set_value(self._texture, self._image)
        for joint in state.joints:
            dpg.set_value(self._readings[joint.name], f"{math.degrees(joint.position):7.1f}")
        dpg.render_dearpygui_frame()
        dpg.run_callbacks(dpg.get_callback_queue())
        if not dpg.is_dearpygui_running():
            return None
        changed, self._changed = self._changed, {}
        return changed

    def close(self) -> None:
        dpg.destroy_context()

    def _on_slider(self, _sender: object, degrees: float, name: str) -> None:
        self._changed[name] = math.radians(degrees)

    def _on_lift(self, _sender: object, degrees: float) -> None:
        self._set({name: sense * degrees for name, sense in _LIFT.items()})

    def _on_rest(self) -> None:
        self._set(dict.fromkeys(self._sliders, 0.0))
        dpg.set_value(self._lift, 0.0)

    def _set(self, targets_deg: Mapping[str, float]) -> None:
        """Set targets as if their sliders had been moved."""
        for name, degrees in targets_deg.items():
            dpg.set_value(self._sliders[name], degrees)
            self._changed[name] = math.radians(degrees)
