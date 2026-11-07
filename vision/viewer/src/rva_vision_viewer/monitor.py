from collections.abc import Callable
from typing import Final

from rva_schemas.attention import AttentionState, AttentionTarget
from rva_schemas.record import LiveFrame, Record
from rva_vision_viewer.overlay import draw
from rva_vision_viewer.stream import RtspPublisher, VideoStreamConfig
from rva_vision_viewer.window import DebugWindow


class CameraMonitor:
    """`Recorder` that shows the camera as it is processed, in a debug window, over RTSP, or both.

    Each live frame is drawn over with the people seen in it and the point attended to, from the
    last attention recorded, then shown. It consumes records like any other recorder: nothing of
    what it shows is routed back to the camera pipeline. Live frames come in the thread that
    records them, the camera's, which the window lives in; the attention may come from any
    thread.
    """

    def __init__(
        self,
        *,
        window: tuple[int, int] | None,
        stream: VideoStreamConfig | None,
        on_key: Callable[[str], None] | None,
    ) -> None:
        """Set up; the window opens and the stream starts with the first frame.

        Args:
            window: The debug window's size for the camera image, or `None` for no window.
            stream: Where to publish the drawn frames, or `None` not to.
            on_key: Called in the camera's thread with each key pressed in the window; `q` once
                the window is closed, by the key or otherwise.
        """
        self._window_size = window
        self._stream: Final = stream
        self._on_key: Final = on_key
        self._window: DebugWindow | None = None
        self._publisher: RtspPublisher | None = None
        self._attention: AttentionTarget | None = None

    def record(self, entry: Record) -> None:
        match entry:
            case LiveFrame():
                self._show(entry)
            case AttentionState():
                self._attention = entry.target
            case _:
                return

    def _show(self, live: LiveFrame) -> None:
        draw(live.frame, live.people, live.timings, self._attention)
        if self._stream is not None and self._publisher is None:
            height, width = live.frame.shape[:2]
            self._publisher = RtspPublisher(self._stream, width, height)
        if self._publisher is not None:
            self._publisher.send(live.frame)
        if self._window_size is None:
            return
        if self._window is None:
            self._window = DebugWindow(*self._window_size)
        key = self._window.show(live.frame, live.people, live.depth, live.main_user, live.timings)
        if key == "q":
            # Closed by the user: it stays closed while the camera goes on.
            self._window.close()
            self._window = None
            self._window_size = None
        if key is not None and self._on_key is not None:
            self._on_key(key)

    def close(self) -> None:
        """Close the window and stop streaming, once the last frame was shown."""
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        if self._window is not None:
            self._window.close()
            self._window = None
