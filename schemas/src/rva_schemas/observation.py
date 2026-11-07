from rva_pydantic import RVABaseModel
from rva_schemas.record import PeopleInView


class VisualObservation(RVABaseModel):
    """What the camera saw in one frame, as the visual attention takes it."""

    time: float
    """When the frame was taken, in seconds of a monotonic clock. Every dynamic of the attention
    runs on these times, never on the wall clock, so a replay gives the same result."""
    people: PeopleInView
    main_user: int | None
    """Track of the main user, whose skeleton and hands are placed in 3D; `None` without one."""
