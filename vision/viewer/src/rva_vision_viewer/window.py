import string
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final, assert_never

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt

from rva_schemas.attention import AttentionEntity, AttentionState, FaceEntity, HandEntity
from rva_schemas.interaction import PersonEngagement
from rva_schemas.record import PeopleInView, TrackedPerson
from rva_schemas.vision import Body, DepthMap, Hand, Keypoint, MainUser, VisionTimings
from rva_vision_viewer.overlay import OverlayOptions

_TITLE: Final = "lune see"
_PANEL_WIDTH: Final = 320
_MARGIN: Final = 40
"""Room taken around the image and the panel by the window's padding."""
_DEPTH_SCALE: Final = 0.5
"""Size of the depth image, relative to the camera's."""
_MAX_DEPTH_M: Final = 4.0
"""Depth at the far end of the colormap; farther is drawn the same."""
_PROBE_RADIUS: Final = 2
"""Half the side of the square of depth pixels whose median is shown, in pixels."""
HISTORY_S: Final = 30.0
"""How much of each person's signals and openness, and of the sound, is plotted."""
_SOUND_HEIGHT: Final = 110
"""Height of the sound's plot, between the camera and the people's rows."""
_ROW_HEIGHT: Final = 130
"""Height of a person's row of plots, under the camera."""
_ROWS_SHOWN: Final = 3
"""How many people's rows fit under the camera before they scroll."""
_ABOUT_WIDTH: Final = 300
"""Width of what is written about a person, left of their plots."""
_OPENNESS_WIDTH: Final = 520
_LANE_HEIGHT: Final = 0.8
"""Height of a signal when on, in lanes: the gap keeps lanes apart."""
_SCORE: Final = "engagement"
"""The lane of the engagement score, a line from 0 at its bottom to 1 at its top."""
_OPENNESS: Final = ("left eye", "right eye", "mouth")
_PROBE_HEIGHT: Final = 30
"""Room the depth's caption takes between the depth and the attention's plot."""
_MIN_ATTENTION_HEIGHT: Final = 80
# RGB
_KNOWN: Final = (80, 200, 80)
_UNKNOWN: Final = (255, 165, 0)
_UNCONFIRMED: Final = (160, 160, 160)
_KEYS: Final = {
    getattr(dpg, f"mvKey_{letter}"): letter.lower() for letter in string.ascii_uppercase
}


class DebugWindow:
    """A Dear PyGui window showing the camera pipeline, for debugging.

    The camera image and, right of it, its depth once some is given, with the distance under the
    mouse, and under it the attention: the score of every face and hand that could be attended
    to, a curve each, the one attended to marked. Then the timings and checkboxes for what is
    drawn over the camera image. Underneath,
    across the window, a row per person in view: their gaze, expression and skeleton, a timeline
    of their signals, e.g. whether they look at the robot, one lane each, and a plot of how open
    their eyes and mouth have been. The rows share their time axis, so who speaks when reads
    down them. Lives in the thread that creates it, which also shows the frames and closes it.
    """

    def __init__(self, width: int, height: int) -> None:
        """Open the window, with the camera image scaled down to `width` by `height`."""
        self._size: Final = (width, height)
        self._image: Final = np.zeros((height, width, 3), np.float32)
        self._depth_size: Final = (round(width * _DEPTH_SCALE), round(height * _DEPTH_SCALE))
        self._depth_image: Final = np.zeros((*reversed(self._depth_size), 3), np.float32)
        self._keys: Final = deque[str]()
        self._panels: Final[dict[int, _PersonPanel]] = {}
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
                with dpg.group():
                    with dpg.group(show=False) as depth_view:
                        self._depth_view: Final = depth_view
                        self._depth_picture: Final = dpg.add_image(self._depth_texture)
                        self._probe: Final = dpg.add_text(wrap=0)
                    self._attention: Final = _AttentionPlot()
                with dpg.group(width=_PANEL_WIDTH):
                    self._timings: Final = dpg.add_text()
                    dpg.add_text("Drawn over the camera:")
                    self._bodies: Final = dpg.add_checkbox(label="skeletons", default_value=True)
                    self._hands: Final = dpg.add_checkbox(label="hands", default_value=True)
                    self._faces: Final = dpg.add_checkbox(label="faces", default_value=True)
                    self._engagement: Final = dpg.add_checkbox(
                        label="engagement", default_value=True
                    )
                    self._main_user: Final = dpg.add_text(wrap=_PANEL_WIDTH)
                    self._unmatched: Final = dpg.add_text(color=_UNCONFIRMED)
            self._sound: Final = _SoundPlot()
            with dpg.child_window(height=_ROW_HEIGHT, border=False) as people:
                self._people: Final = people
        dpg.create_viewport(
            title=_TITLE,
            width=round(width * (1 + _DEPTH_SCALE)) + _PANEL_WIDTH + 2 * _MARGIN,
            height=height + _SOUND_HEIGHT + _ROWS_SHOWN * _ROW_HEIGHT + _MARGIN,
            # Drawing must not wait for the screen: it would slow the camera pipeline down.
            vsync=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(root, True)

    def options(self) -> OverlayOptions:
        """What the checkboxes ask to be drawn over the camera image."""
        return OverlayOptions(
            bodies=bool(dpg.get_value(self._bodies)),
            hands=bool(dpg.get_value(self._hands)),
            faces=bool(dpg.get_value(self._faces)),
            engagement=bool(dpg.get_value(self._engagement)),
        )

    def show(
        self,
        frame: npt.NDArray[np.uint8],
        people: PeopleInView,
        depth: DepthMap | None,
        main_user: MainUser | None,
        timings: VisionTimings,
        engagement: Mapping[int, PersonEngagement],
        attention: AttentionState | None,
        sound: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    ) -> str | None:
        """Show a frame, drawn over already, and what was seen in it.

        `engagement` is how much each person pays attention to the robot, by track id, and
        `attention` what could be attended to in the frame; `None` until some is known. `sound`
        is the times, from `time.monotonic`, and the values, from 0 to 1, of each series to plot
        of the sound lately, by name: nothing is plotted while they are all empty.

        Returns:
            A key pressed since the last frame, `q` once the window is closed, else `None`.
        """
        if not dpg.is_dearpygui_running():
            return "q"
        small = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        np.multiply(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self._image)
        dpg.set_value(self._texture, self._image)
        # The camera shares the width with the column of its depth and the attention, which is
        # at the depth's smaller scale.
        space_width = dpg.get_viewport_client_width() - _PANEL_WIDTH - 2 * _MARGIN
        space_width = round(space_width / (1 + _DEPTH_SCALE))
        # The people's rows take what they need under the camera, up to a few; more scroll.
        rows_height = min(max(len(people.people), 1), _ROWS_SHOWN) * _ROW_HEIGHT
        dpg.configure_item(self._people, height=rows_height)
        now = time.monotonic() - self._start
        sound_height = _SOUND_HEIGHT if self._sound.update(now, sound, self._start) else 0
        space_height = dpg.get_viewport_client_height() - rows_height - sound_height - _MARGIN
        width, height = fit(self._size, (space_width, space_height))
        dpg.configure_item(self._picture, width=width, height=height)
        dpg.configure_item(self._depth_view, show=depth is not None)
        side_width, depth_height = round(width * _DEPTH_SCALE), round(height * _DEPTH_SCALE)
        if depth is not None:
            self._show_depth(depth, (side_width, depth_height))
        # The attention takes what the depth leaves of the camera's height.
        taken = 0 if depth is None else depth_height + _PROBE_HEIGHT
        self._attention.update(
            now, attention, (side_width, max(height - taken, _MIN_ATTENTION_HEIGHT))
        )
        dpg.set_value(self._timings, describe_timings(timings))
        self._show_people(people, engagement)
        dpg.set_value(self._main_user, describe_main_user(main_user))
        dpg.configure_item(
            self._main_user,
            color=_KNOWN if main_user is not None and main_user.eye_contact else _UNCONFIRMED,
        )
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

    def _show_people(
        self, people: PeopleInView, engagement: Mapping[int, PersonEngagement]
    ) -> None:
        """Bring each person's panel up to date, adding the newcomers' and removing the gone's."""
        now = time.monotonic() - self._start
        for person in people.people:
            panel = self._panels.get(person.track_id)
            if panel is None:
                panel = self._panels[person.track_id] = _PersonPanel(self._people)
            panel.update(now, person, engagement.get(person.track_id))
        for track_id in self._panels.keys() - {person.track_id for person in people.people}:
            self._panels.pop(track_id).delete()
        count = len(people.unmatched_bodies)
        dpg.set_value(self._unmatched, f"{count} skeletons without a face" if count else "")


class _AttentionPlot:
    """The score of everything that could be attended to, a curve each, over the last while."""

    def __init__(self) -> None:
        self._scores: Final[dict[str, deque[tuple[float, float]]]] = {}
        self._lines: Final[dict[str, int | str]] = {}
        with dpg.plot(label="Attention", height=_MIN_ATTENTION_HEIGHT) as plot:
            self._plot: Final = plot
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            self._time: Final = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
            self._axis: Final = dpg.add_plot_axis(dpg.mvYAxis)

    def update(self, now: float, attention: AttentionState | None, size: tuple[int, int]) -> None:
        """Add the candidates' scores; a candidate gone leaves once its curve has scrolled out."""
        dpg.configure_item(self._plot, width=size[0], height=size[1])
        candidates = () if attention is None else attention.candidates
        target = None if attention is None or attention.target is None else attention.target.entity
        for candidate in candidates:
            name = candidate_name(candidate.entity)
            self._scores.setdefault(name, deque()).append((now, candidate.score))
        attended = None if target is None else candidate_name(target)
        for name, history in list(self._scores.items()):
            _trim(history, now)
            if not history:
                del self._scores[name]
                dpg.delete_item(self._lines.pop(name))
                continue
            data = [[t for t, _ in history], [score for _, score in history]]
            if name not in self._lines:
                self._lines[name] = dpg.add_line_series(*data, parent=self._axis)
            dpg.set_value(self._lines[name], data)
            dpg.configure_item(self._lines[name], label=f"> {name}" if name == attended else name)
        dpg.set_axis_limits(self._time, now - HISTORY_S, now)
        dpg.fit_axis_data(self._axis)


class _SoundPlot:
    """How loud the microphone and the robot's voice were, and how likely speech was heard."""

    def __init__(self) -> None:
        self._lines: Final[dict[str, int | str]] = {}
        with dpg.plot(height=_SOUND_HEIGHT - 8, width=-1, show=False) as plot:
            self._plot: Final = plot
            dpg.add_plot_legend(outside=True, location=dpg.mvPlot_Location_East)
            self._time: Final = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
            self._axis: Final = dpg.add_plot_axis(dpg.mvYAxis)
            dpg.set_axis_limits(self._axis, 0, 1)

    def update(
        self,
        now: float,
        sound: Mapping[str, tuple[Sequence[float], Sequence[float]]],
        start: float,
    ) -> bool:
        """Plot each series, whose times are `start` later than the window's `now`.

        Returns:
            Whether the plot shows: once there is sound, e.g. not with the camera alone.
        """
        for name, (times, values) in sound.items():
            if name not in self._lines:
                self._lines[name] = dpg.add_line_series([0.0], [0.0], label=name, parent=self._axis)
            dpg.set_value(self._lines[name], [[time - start for time in times], list(values)])
        dpg.set_axis_limits(self._time, now - HISTORY_S, now)
        shown = any(times for times, _ in sound.values())
        dpg.configure_item(self._plot, show=shown)
        return shown


class _PersonPanel:
    """What the window shows of one person, for as long as they are tracked."""

    def __init__(self, parent: int | str) -> None:
        self._signals: Final[dict[str, deque[tuple[float, bool]]]] = {
            name: deque() for name in signals(None, None)
        }
        self._score: Final = deque[tuple[float, float]]()
        self._openness: Final = deque[tuple[float, tuple[float, float, float]]]()
        lanes = [*self._signals, _SCORE]
        with dpg.group(parent=parent, horizontal=True) as group:
            self._group: Final = group
            plots_height = _ROW_HEIGHT - 8
            with dpg.child_window(width=_ABOUT_WIDTH, height=plots_height, border=False):
                self._title: Final = dpg.add_text(wrap=0)
                self._details: Final = dpg.add_text(wrap=0)
            # A negative width leaves that much room on the right, for the other plot.
            with dpg.plot(height=plots_height, width=-_OPENNESS_WIDTH - 16):
                self._signals_time: Final = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
                axis = dpg.add_plot_axis(dpg.mvYAxis)
                dpg.set_axis_ticks(
                    axis, tuple((name, lane + _LANE_HEIGHT / 2) for lane, name in enumerate(lanes))
                )
                dpg.set_axis_limits(axis, -0.1, len(lanes))
                self._lanes: Final = {
                    name: dpg.add_shade_series([0.0], [float(lane)], y2=[float(lane)], parent=axis)
                    for lane, name in enumerate(self._signals)
                }
                self._score_line: Final = dpg.add_line_series(
                    [0.0], [float(len(self._signals))], parent=axis
                )
            with dpg.plot(height=plots_height, width=-1):
                dpg.add_plot_legend(outside=True, location=dpg.mvPlot_Location_East)
                self._openness_time: Final = dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
                axis = dpg.add_plot_axis(dpg.mvYAxis)
                dpg.set_axis_limits(axis, 0, 1)
                self._openness_lines: Final = [
                    dpg.add_line_series([0.0], [0.0], label=name, parent=axis) for name in _OPENNESS
                ]

    def update(
        self, now: float, person: TrackedPerson, engagement: PersonEngagement | None
    ) -> None:
        dpg.set_value(self._title, title(person))
        dpg.configure_item(self._title, color=_color(person))
        dpg.set_value(self._details, describe_person(person, engagement))
        for lane, (name, on) in enumerate(signals(person, engagement).items()):
            history = self._signals[name]
            history.append((now, on))
            _trim(history, now)
            times = [t for t, _ in history]
            tops = [lane + _LANE_HEIGHT * value for _, value in history]
            dpg.set_value(self._lanes[name], [times, tops, [float(lane)] * len(times)])
        if engagement is not None:
            self._score.append((now, engagement.score))
        _trim(self._score, now)
        bottom = len(self._signals)
        dpg.set_value(
            self._score_line,
            [[t for t, _ in self._score], [bottom + _LANE_HEIGHT * v for _, v in self._score]],
        )
        if person.face is not None:
            face = person.face
            self._openness.append((now, (face.left_eye_open, face.right_eye_open, face.mouth_open)))
        _trim(self._openness, now)
        times = [t for t, _ in self._openness]
        for index, line in enumerate(self._openness_lines):
            dpg.set_value(line, [times, [values[index] for _, values in self._openness]])
        for axis in (self._signals_time, self._openness_time):
            dpg.set_axis_limits(axis, now - HISTORY_S, now)

    def delete(self) -> None:
        dpg.delete_item(self._group)


def _trim[T](history: deque[tuple[float, T]], now: float) -> None:
    while history and history[0][0] < now - HISTORY_S:
        history.popleft()


def signals(person: TrackedPerson | None, engagement: PersonEngagement | None) -> dict[str, bool]:
    """A person's signals shown on their timeline; each off for nobody, or while unknown."""
    return {
        "looking": engagement is not None and engagement.looking,
        "speaking": person is not None and person.speaker is not None and person.speaker.speaking,
        "engaged": engagement is not None and engagement.engaged,
    }


def candidate_name(entity: AttentionEntity) -> str:
    """What a face or hand that could be attended to is called on the attention's plot."""
    match entity:
        case FaceEntity():
            return f"face #{entity.track_id}"
        case HandEntity():
            return f"{entity.side} hand #{entity.track_id}"
        case _:
            assert_never(entity)


def describe_person(person: TrackedPerson, engagement: PersonEngagement | None) -> str:
    """A person's gaze and expression, their skeleton and their engagement, a line each."""
    lines = []
    if person.face is not None:
        face = person.face
        lines.append(
            f"gaze {face.gaze.yaw:+.0f} {face.gaze.pitch:+.0f} deg   "
            f"{face.expression} {face.expression_score:.2f}"
        )
    if person.body is not None:
        lines.append(describe_body(person.body))
    if engagement is not None:
        lines.append(
            f"engagement {engagement.score:.2f}: closeness {engagement.closeness:.2f}, "
            f"staying {engagement.staying:.2f}"
        )
    return "\n".join(lines)


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
