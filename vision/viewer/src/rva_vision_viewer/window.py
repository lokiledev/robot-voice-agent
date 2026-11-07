import string
import time
from collections import deque
from typing import Final

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt

from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import Body, DepthMap, FaceFeatures, Hand, Keypoint, MainUser, VisionTimings

_TITLE: Final = "lune see"
_PANEL_WIDTH: Final = 320
_MARGIN: Final = 40
"""Room taken around the image and the panel by the window's padding."""
_PLOT_S: Final = 10.0
"""How much of each mouth's history is plotted."""
_DEPTH_SCALE: Final = 0.5
"""Size of the depth image, relative to the camera's."""
_MAX_DEPTH_M: Final = 4.0
"""Depth at the far end of the colormap; farther is drawn the same."""
_PROBE_RADIUS: Final = 2
"""Half the side of the square of depth pixels whose median is shown, in pixels."""
_TIMELINE_S: Final = 30.0
"""How much of the main user's signals the timeline shows."""
_TIMELINE_HEIGHT: Final = 120
_LANE_HEIGHT: Final = 0.8
"""Height of a signal when on, in lanes: the gap keeps lanes apart."""
# RGB
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)
_KEYS: Final = {
    getattr(dpg, f"mvKey_{letter}"): letter.lower() for letter in string.ascii_uppercase
}


class DebugWindow:
    """A Dear PyGui window showing the camera pipeline, for debugging.

    The camera image, and beside it each person in view with their eyes, mouth, gaze, expression
    and skeleton, and a plot of how open their mouths have been. Right of the camera image, its depth
    once some is given, with the distance under the mouse. Underneath, a timeline of the main user's
    signals, e.g. eye contact, one lane each. Lives in the thread that creates it, which also shows
    the frames and closes it.
    """

    def __init__(self, width: int, height: int) -> None:
        """Open the window, with the camera image scaled down to `width` by `height`."""
        self._size: Final = (width, height)
        self._image: Final = np.zeros((height, width, 3), np.float32)
        self._depth_size: Final = (round(width * _DEPTH_SCALE), round(height * _DEPTH_SCALE))
        self._depth_image: Final = np.zeros((*reversed(self._depth_size), 3), np.float32)
        self._keys: Final = deque[str]()
        self._mouths: Final[dict[int, deque[tuple[float, float]]]] = {}
        self._series: Final[dict[int, int | str]] = {}
        self._signals: Final[dict[str, deque[tuple[float, bool]]]] = {}
        self._lanes: Final[dict[str, int | str]] = {}
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
            self._depth_texture: Final = dpg.add_raw_texture(
                *self._depth_size,
                self._depth_image,  # ty: ignore[invalid-argument-type]  # Any buffer is read; the stubs only list lists.
                format=dpg.mvFormat_Float_rgb,
            )
        with dpg.handler_registry():
            for key, letter in _KEYS.items():
                dpg.add_key_press_handler(key, callback=self._on_key, user_data=letter)
        with dpg.window() as root:
            with dpg.group(horizontal=True):
                self._picture: Final = dpg.add_image(self._texture)
                with dpg.group(show=False) as depth_view:
                    self._depth_view: Final = depth_view
                    self._depth_picture: Final = dpg.add_image(self._depth_texture)
                    self._probe: Final = dpg.add_text(wrap=0)
                # A negative height leaves that much room below, for the timeline.
                with dpg.child_window(width=_PANEL_WIDTH, height=-_TIMELINE_HEIGHT - _MARGIN // 4):
                    self._timings: Final = dpg.add_text()
                    self._main_user: Final = dpg.add_text(wrap=0)
                    self._people: Final = dpg.add_group()
                    with dpg.plot(label="Mouth open", height=220, width=-1):
                        dpg.add_plot_legend()
                        self._time_axis: Final = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                        self._mouth_axis: Final = dpg.add_plot_axis(dpg.mvYAxis)
                        dpg.set_axis_limits(self._mouth_axis, 0, 1)
            with dpg.plot(label="Main user", height=_TIMELINE_HEIGHT, width=-1):
                self._timeline_axis: Final = dpg.add_plot_axis(dpg.mvXAxis, label="s")
                self._lane_axis: Final = dpg.add_plot_axis(dpg.mvYAxis)
        dpg.create_viewport(
            title=_TITLE,
            width=round(width * (1 + _DEPTH_SCALE)) + _PANEL_WIDTH + 2 * _MARGIN,
            height=height + _TIMELINE_HEIGHT + _MARGIN,
            # Drawing must not wait for the screen: it would slow the camera pipeline down.
            vsync=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(root, True)

    def show(
        self,
        frame: npt.NDArray[np.uint8],
        people: PeopleInView,
        depth: DepthMap | None,
        main_user: MainUser | None,
        timings: VisionTimings,
    ) -> str | None:
        if not dpg.is_dearpygui_running():
            return "q"
        small = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        np.multiply(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self._image)
        dpg.set_value(self._texture, self._image)
        space_width = dpg.get_viewport_client_width() - _PANEL_WIDTH - _MARGIN
        if depth is not None:
            # The camera and its depth share the width, the depth at its smaller scale.
            space_width = round((space_width - _MARGIN) / (1 + _DEPTH_SCALE))
        space_height = dpg.get_viewport_client_height() - _TIMELINE_HEIGHT - _MARGIN
        width, height = fit(self._size, (space_width, space_height))
        dpg.configure_item(self._picture, width=width, height=height)
        dpg.configure_item(self._depth_view, show=depth is not None)
        if depth is not None:
            self._show_depth(depth, (round(width * _DEPTH_SCALE), round(height * _DEPTH_SCALE)))
        dpg.set_value(self._timings, describe_timings(timings))
        self._show_people(people)
        self._plot_mouths(people)
        dpg.set_value(self._main_user, describe_main_user(main_user))
        dpg.configure_item(
            self._main_user,
            color=_KNOWN if main_user is not None and main_user.eye_contact else _UNCONFIRMED,
        )
        self._plot_signals(signals(main_user))
        dpg.render_dearpygui_frame()
        dpg.run_callbacks(dpg.get_callback_queue())
        if not dpg.is_dearpygui_running():
            return "q"
        return self._keys.popleft() if self._keys else None

    def close(self) -> None:
        dpg.destroy_context()

    def _on_key(self, sender: int | str, app_data: int, user_data: str) -> None:
        self._keys.append(user_data)

    def _show_depth(self, depth: DepthMap, size: tuple[int, int]) -> None:
        """Color the depth, near in blue and far in red, and give the distance under the mouse."""
        small = cv2.resize(depth.metres, self._depth_size, interpolation=cv2.INTER_NEAREST)
        levels = np.clip(small / _MAX_DEPTH_M * 255, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
        np.multiply(
            cv2.cvtColor(colors, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self._depth_image
        )
        dpg.set_value(self._depth_texture, self._depth_image)
        dpg.configure_item(self._depth_picture, width=size[0], height=size[1])
        where, point = "at the center", (0.5, 0.5)
        for picture in (self._picture, self._depth_picture):
            if dpg.is_item_hovered(picture) is True:
                mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
                left, top = dpg.get_item_rect_min(picture)
                picture_width, picture_height = dpg.get_item_rect_size(picture)
                where = "under the mouse"
                point = ((mouse_x - left) / picture_width, (mouse_y - top) / picture_height)
        dpg.set_value(
            self._probe,
            f"{describe_depth(depth, point)} {where}   colors 0 to {_MAX_DEPTH_M:.0f} m",
        )
        dpg.configure_item(self._probe, color=_KNOWN if depth.calibrated else _UNKNOWN)

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

    def _plot_signals(self, values: dict[str, bool]) -> None:
        """Add a value to each signal's lane, shaded while on, and scroll the timeline."""
        now = time.monotonic() - self._start
        for name, value in values.items():
            history = self._signals.setdefault(name, deque())
            history.append((now, value))
            while history[0][0] < now - _TIMELINE_S:
                history.popleft()
            lane = list(self._signals).index(name)
            times = [t for t, _ in history]
            tops = [lane + _LANE_HEIGHT * on for _, on in history]
            if name in self._lanes:
                dpg.set_value(self._lanes[name], [times, tops, [lane] * len(times)])
                continue
            self._lanes[name] = dpg.add_shade_series(
                times, tops, y2=[lane] * len(times), parent=self._lane_axis
            )
            dpg.set_axis_ticks(
                self._lane_axis,
                tuple((label, index + _LANE_HEIGHT / 2) for index, label in enumerate(self._lanes)),
            )
            dpg.set_axis_limits(self._lane_axis, -0.1, len(self._lanes))
        dpg.set_axis_limits(self._timeline_axis, now - _TIMELINE_S, now)


def signals(main_user: MainUser | None) -> dict[str, bool]:
    """The main user's signals shown on the timeline, each off while nobody is in view."""
    return {
        "eye contact": main_user is not None and main_user.eye_contact,
        "lips moving": main_user is not None and main_user.lips_moving,
        "speaking": main_user is not None
        and main_user.active_speaker is not None
        and main_user.active_speaker.speaking,
    }


def describe_main_user(main_user: MainUser | None) -> str:
    if main_user is None:
        return "nobody in view"
    contact = "eye contact" if main_user.eye_contact else "no eye contact"
    lips = "lips moving" if main_user.lips_moving else "lips still"
    match main_user.active_speaker:
        case None:
            speaking = "no microphone"
        case speaker:
            speaking = (
                f"{'speaking' if speaker.speaking else 'not speaking'} ({speaker.score:+.1f})"
            )
    return (
        f"main user #{main_user.track_id}: {contact}, gaze {main_user.gaze_off_camera:.0f} deg off"
        f"; {lips}, motion {main_user.mouth_motion:.3f}; {speaking}"
    )


def fit(image: tuple[int, int], space: tuple[int, int]) -> tuple[int, int]:
    """The largest size with the image's aspect ratio that fits in `space`, at least 1x1."""
    scale = max(min(space[0] / image[0], space[1] / image[1]), 0.0)
    return max(round(image[0] * scale), 1), max(round(image[1] * scale), 1)


def depth_at(depth: npt.NDArray[np.float32], point: tuple[float, float]) -> float:
    """Median depth in a small square around a point given as fractions of the width and height."""
    height, width = depth.shape
    x = min(max(round(point[0] * (width - 1)), 0), width - 1)
    y = min(max(round(point[1] * (height - 1)), 0), height - 1)
    patch = depth[
        max(y - _PROBE_RADIUS, 0) : y + _PROBE_RADIUS + 1,
        max(x - _PROBE_RADIUS, 0) : x + _PROBE_RADIUS + 1,
    ]
    return float(np.median(patch))


def describe_depth(depth: DepthMap, point: tuple[float, float]) -> str:
    """The depth at a point, and whether the camera was calibrated to measure it."""
    calibration = "calibrated" if depth.calibrated else "UNCALIBRATED"
    return f"depth {depth_at(depth.metres, point):.2f} m ({calibration})"


def title(person: TrackedPerson) -> str:
    who = (
        "unknown"
        if person.person is None
        else f"{person.person.name} ({str(person.person.user_id)[:8]})"
    )
    said = "" if person.said is None else f': "{person.said}"'
    return f"#{person.track_id} {who}{said}"


def describe_body(body: Body) -> str:
    """How many keypoints of the body and of each hand were found; `-` for a hand not found."""

    def found(keypoints: tuple[Keypoint | None, ...]) -> str:
        return f"{sum(point is not None for point in keypoints)}/{len(keypoints)}"

    def hand(found_hand: Hand | None) -> str:
        return "-" if found_hand is None else found(found_hand.keypoints)

    return (
        f"skeleton {found(body.keypoints)}, hands left {hand(body.left_hand)} "
        f"right {hand(body.right_hand)}"
    )


def describe_timings(timings: VisionTimings) -> str:
    return (
        f"{timings.fps:.0f} fps\n"
        f"capture {timings.capture_ms:.1f} ms  detect {timings.detect_ms:.1f} ms\n"
        f"embed {timings.embed_ms:.1f} ms  faces {timings.faces_ms:.1f} ms\n"
        f"bodies {timings.bodies_ms:.1f} ms  depth {timings.depth_ms:.1f} ms"
        f"\nspeaker {timings.speaker_ms:.1f} ms"
    )


def _color(person: TrackedPerson) -> tuple[int, int, int]:
    if not person.confirmed:
        return _UNCONFIRMED
    return _UNKNOWN if person.person is None else _KNOWN
