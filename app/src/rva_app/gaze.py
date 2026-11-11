"""The attention's target, sent to the head controller for the head camera to look at."""

from typing import Final

from loguru import logger

from rva_app.config import RuntimeConfig
from rva_robot_client.head import HeadSender, open_gaze_sender
from rva_robot_schemas.head import GazeTarget
from rva_schemas.attention import AttentionState


class GazeOut:
    """Sends each target the attention has to the head controller, whenever that runs.

    The controller is looked for again at each target until it is found, and after it stops.
    """

    def __init__(self, namespace: str) -> None:
        self._namespace: Final = namespace
        self._sender: HeadSender[GazeTarget] | None = None

    def send(self, state: AttentionState) -> None:
        """Send the state's target, with the time its frame was taken; nothing without one."""
        if state.target is None:
            return
        if self._sender is None:
            self._sender = open_gaze_sender(self._namespace, "lune_attention")
            if self._sender is None:
                return
            logger.info("Sending the attention's targets to the head controller")
        point = state.target.point
        try:
            self._sender.send(GazeTarget(timestamp=state.time, x=point.x, y=point.y, z=point.z))
        except BlockingIOError:
            # Its mailbox filled up: that controller stopped, and another may have started since.
            logger.warning("The head controller stopped taking gaze targets")
            self._sender = None


def gaze_out(config: RuntimeConfig) -> GazeOut | None:
    """Where the attention's targets go for `config`'s profile: nowhere without a head controller."""
    if config.robot is None or config.head_controller is None:
        return None
    return GazeOut(config.robot.namespace)
