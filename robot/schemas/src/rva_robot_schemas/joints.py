from rva_pydantic import RVABaseModel


class JointReading(RVABaseModel):
    name: str
    """As the robot's description names the joint, e.g. `yaw_body`."""
    position: float
    """In radians."""
    velocity: float
    """In radians per second."""


class JointState(RVABaseModel):
    """Every actuated joint of the robot, at one instant."""

    timestamp: float
    """When the joints were read, in seconds of `time.monotonic()`, which on Linux is the same clock
    in every process of the machine."""
    joints: tuple[JointReading, ...]


class JointTarget(RVABaseModel):
    name: str
    position: float
    """In radians; the robot clamps it to the joint's range."""


class JointCommand(RVABaseModel):
    """Positions for some of the joints to go to; the others keep their last target."""

    targets: tuple[JointTarget, ...]
