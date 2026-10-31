from uuid import UUID


class BodyBusyError(Exception):
    """The body refused a motion because it is already doing something."""

    def __init__(self, busy_with: str) -> None:
        super().__init__(f"Busy: {busy_with}")
        self.busy_with = busy_with


class UnknownMotionError(Exception):
    def __init__(self, motion_id: UUID) -> None:
        super().__init__(f"No motion {motion_id}")
        self.motion_id = motion_id
