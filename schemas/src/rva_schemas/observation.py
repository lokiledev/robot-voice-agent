from rva_pydantic import RVABaseModel
from rva_schemas.frames import CameraPose
from rva_schemas.record import PeopleInView


class VisualObservation(RVABaseModel):
    """What the camera saw in one frame, as the visual attention and the interaction take it."""

    time: float
    """When the frame was taken, in seconds of a monotonic clock. Every dynamic of the attention
    runs on these times, never on the wall clock, so a replay gives the same result."""
    people: PeopleInView
    camera_pose: CameraPose | None
    """Where the camera was when the frame was taken, to tell its own motion from what it sees;
    `None` for a camera that does not move, or when its pose is not known then."""
