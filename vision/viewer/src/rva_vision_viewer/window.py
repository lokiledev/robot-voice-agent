import string
import time
from collections import deque
from typing import Final

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt

from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import Body, FaceFeatures, Keypoint, VisionTimings

_TITLE: Final = "lune see"
_PANEL_WIDTH: Final = 320
_MARGIN: Final = 40
"""Room taken around the image and the panel by the window's padding."""
_PLOT_S: Final = 10.0
"""How much of each mouth's history is plotted."""
# RGB
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)
_KEYS: Final = {
    getattr(dpg, f"mvKey_{letter}"): letter.lower() for letter in string.ascii_uppercase
}


class DebugWindow:
    """`FrameViewer` in a Dear PyGui window.

    The camera image, and beside it each person in view with their eyes, mouth, gaze, expression
    and skeleton, and a plot of how open their mouths have been. Lives in the thread that creates
    it, which also shows the frames and closes it.
    """

    def __init__(self, width: int, height: int) -> None:
        """Open the window, with the camera image scaled down to `width` by `height`."""
        self._size: Final = (width, height)
        self._image: Final = np.zeros((height, width, 3), np.float32)
        self._keys: Final = deque[str]()
        self._mouths: Final[dict[int, deque[tuple[float, float]]]] = {}
        self._series: Final[dict[int, int | str]] = {}
        self._start: Final = time.monotonic()
        dpg.create_context()
        # Callbacks run when `show` asks, in this thread, rather than in one of Dear PyGui's.
        dpg.configure_app(manual_callback_management=True)
        with dpg.texture_registry():
            self._texture: Final = dpg.add_raw_texture(
                width,
                height,
                self._image,  # ty: ignore[invalid-argument-type]  # Any buffer is read; the stubs only list lists.
                format=dpg.mvFormat_Float_rgb,
            )
        with dpg.handler_registry():
            for key, letter in _KEYS.items():
                dpg.add_key_press_handler(key, callback=self._on_key, user_data=letter)
        with dpg.window() as root, dpg.group(horizontal=True):
            self._picture: Final = dpg.add_image(self._texture)
            with dpg.child_window(width=_PANEL_WIDTH):
                self._timings: Final = dpg.add_text()
                self._people: Final = dpg.add_group()
                with dpg.plot(label="Mouth open", height=220, width=-1):
                    dpg.add_plot_legend()
                    self._time_axis: Final = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                    self._mouth_axis: Final = dpg.add_plot_axis(dpg.mvYAxis)
                    dpg.set_axis_limits(self._mouth_axis, 0, 1)
        dpg.create_viewport(
            title=_TITLE,
            width=width + _PANEL_WIDTH + _MARGIN,
            height=height + _MARGIN,
            # Drawing must not wait for the screen: it would slow the camera pipeline down.
            vsync=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(root, True)

    def show(
        self, frame: npt.NDArray[np.uint8], people: PeopleInView, timings: VisionTimings
    ) -> str | None:
        if not dpg.is_dearpygui_running():
            return "q"
        small = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        np.multiply(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self._image)
        dpg.set_value(self._texture, self._image)
        width, height = fit(
            self._size,
            (
                dpg.get_viewport_client_width() - _PANEL_WIDTH - _MARGIN,
                dpg.get_viewport_client_height() - _MARGIN,
            ),
        )
        dpg.configure_item(self._picture, width=width, height=height)
        dpg.set_value(self._timings, describe_timings(timings))
        self._show_people(people)
        self._plot_mouths(people)
        dpg.render_dearpygui_frame()
        dpg.run_callbacks(dpg.get_callback_queue())
        if not dpg.is_dearpygui_running():
            return "q"
        return self._keys.popleft() if self._keys else None

    def close(self) -> None:
        dpg.destroy_context()

    def _on_key(self, sender: int | str, app_data: int, user_data: str) -> None:
        self._keys.append(user_data)

    def _show_people(self, people: PeopleInView) -> None:
        dpg.delete_item(self._people, children_only=True)
        for person in people.people:
            dpg.add_separator(parent=self._people)
            dpg.add_text(title(person), color=_color(person), parent=self._people, wrap=0)
            if person.face is not None:
                self._show_face(person.face)
            if person.body is not None:
                dpg.add_text(describe_body(person.body), parent=self._people)
        if people.unmatched_bodies:
            dpg.add_separator(parent=self._people)
            dpg.add_text(
                f"{len(people.unmatched_bodies)} skeletons without a face",
                color=_UNCONFIRMED,
                parent=self._people,
            )

    def _show_face(self, face: FaceFeatures) -> None:
        for name, value in (
            ("left eye", face.left_eye_open),
            ("right eye", face.right_eye_open),
            ("mouth", face.mouth_open),
        ):
            dpg.add_progress_bar(
                default_value=value, overlay=f"{name} {value:.2f}", width=-1, parent=self._people
            )
        dpg.add_text(
            f"gaze {face.gaze.yaw:+.0f} {face.gaze.pitch:+.0f} deg   "
            f"{face.expression} {face.expression_score:.2f}",
            parent=self._people,
        )

    def _plot_mouths(self, people: PeopleInView) -> None:
        now = time.monotonic() - self._start
        for person in people.people:
            if person.face is not None:
                self._mouths.setdefault(person.track_id, deque()).append(
                    (now, person.face.mouth_open)
                )
        in_view = {person.track_id for person in people.people}
        for track_id, history in list(self._mouths.items()):
            while history and history[0][0] < now - _PLOT_S:
                history.popleft()
            if track_id in in_view and history:
                times, values = [list(axis) for axis in zip(*history, strict=True)]
                if track_id in self._series:
                    dpg.set_value(self._series[track_id], [times, values])
                else:
                    self._series[track_id] = dpg.add_line_series(
                        times, values, label=f"#{track_id}", parent=self._mouth_axis
                    )
                continue
            del self._mouths[track_id]
            if track_id in self._series:
                dpg.delete_item(self._series.pop(track_id))
        dpg.set_axis_limits(self._time_axis, now - _PLOT_S, now)


def fit(image: tuple[int, int], space: tuple[int, int]) -> tuple[int, int]:
    """The largest size with the image's aspect ratio that fits in `space`, at least 1x1."""
    scale = max(min(space[0] / image[0], space[1] / image[1]), 0.0)
    return max(round(image[0] * scale), 1), max(round(image[1] * scale), 1)


def title(person: TrackedPerson) -> str:
    who = (
        "unknown"
        if person.person is None
        else f"{person.person.name} ({str(person.person.user_id)[:8]})"
    )
    said = "" if person.said is None else f': "{person.said}"'
    return f"#{person.track_id} {who}{said}"


def describe_body(body: Body) -> str:
    """How many keypoints of the body and of each hand were found."""

    def found(keypoints: tuple[Keypoint | None, ...]) -> str:
        return f"{sum(point is not None for point in keypoints)}/{len(keypoints)}"

    return (
        f"skeleton {found(body.keypoints)}, hands {found(body.left_hand.keypoints)} "
        f"{found(body.right_hand.keypoints)}"
    )


def describe_timings(timings: VisionTimings) -> str:
    return (
        f"{timings.fps:.0f} fps\n"
        f"capture {timings.capture_ms:.1f} ms  detect {timings.detect_ms:.1f} ms\n"
        f"embed {timings.embed_ms:.1f} ms  faces {timings.faces_ms:.1f} ms\n"
        f"bodies {timings.bodies_ms:.1f} ms"
    )


def _color(person: TrackedPerson) -> tuple[int, int, int]:
    if not person.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if person.person is None else _KNOWN
