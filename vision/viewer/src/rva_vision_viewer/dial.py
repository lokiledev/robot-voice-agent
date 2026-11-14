"""Where speech is heard from, on a dial seen from above.

The camera's view is a wedge, the voice an arrow from the robot at its centre.
"""

import math
from typing import Final

import dearpygui.dearpygui as dpg

from rva_schemas.vision import CameraIntrinsics

_SIZE: Final = 160
"""Side of the dial, in pixels."""
_MARGIN: Final = 12
HELD_S: Final = 0.3
"""How long a direction heard is shown: longer than between two, at 20 per second."""
# RGBA
_RIM: Final = (120, 120, 120, 255)
_VIEW: Final = (90, 140, 200, 255)
_VOICE: Final = (80, 200, 80, 255)


def dial_point(bearing_deg: float, radius: float) -> tuple[float, float]:
    """Where a bearing, in degrees right of the camera's axis, reaches `radius` from the centre.

    Up the dial is ahead of the camera, its right the camera's right; y grows down, as on screen.
    """
    bearing = math.radians(bearing_deg)
    return radius * math.sin(bearing), -radius * math.cos(bearing)


def view_edges(intrinsics: CameraIntrinsics) -> tuple[float, float]:
    """The bearings of the image's left and right edges, in degrees right of the camera's axis."""
    return (
        math.degrees(math.atan2(-intrinsics.cx, intrinsics.fx)),
        math.degrees(math.atan2(intrinsics.width - intrinsics.cx, intrinsics.fx)),
    )


def describe(bearing_deg: float) -> str:
    side = "right" if bearing_deg >= 0 else "left"
    return f"Speech {abs(bearing_deg):.0f}° {side} of the camera"


class DoaDial:
    """The dial, in the Dear PyGui container being built; hidden until a direction is heard."""

    def __init__(self) -> None:
        with dpg.group(show=False) as group:
            self._group: Final = group
            self._label: Final = dpg.add_text()
            self._drawing: Final = dpg.add_drawlist(width=_SIZE, height=_SIZE)

    def update(
        self, heard: tuple[float, float] | None, intrinsics: CameraIntrinsics | None, now: float
    ) -> None:
        """Draw the last direction heard, as its bearing and when, on the clock of `now`.

        An arrow while it is recent; the rim alone once speech has not been heard for a moment.
        """
        if heard is None:
            return
        dpg.configure_item(self._group, show=True)
        dpg.delete_item(self._drawing, children_only=True)
        centre = _SIZE / 2
        radius = centre - _MARGIN
        dpg.draw_circle((centre, centre), radius, color=_RIM, parent=self._drawing)
        if intrinsics is not None:
            for edge in view_edges(intrinsics):
                x, y = dial_point(edge, radius)
                dpg.draw_line(
                    (centre, centre), (centre + x, centre + y), color=_VIEW, parent=self._drawing
                )
        bearing, when = heard
        if now - when > HELD_S:
            dpg.set_value(self._label, "No speech heard")
            return
        x, y = dial_point(bearing, radius)
        dpg.draw_arrow(
            (centre + x, centre + y),
            (centre, centre),
            color=_VOICE,
            thickness=3,
            parent=self._drawing,
        )
        dpg.set_value(self._label, describe(bearing))
