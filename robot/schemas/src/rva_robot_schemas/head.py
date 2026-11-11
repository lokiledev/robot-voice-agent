from rva_pydantic import RVABaseModel


class HeadPose(RVABaseModel):
    """The pose of the head frame, the MJCF's `head` site, in the robot's base frame.

    The base frame is the MJCF's root: on the table top, under the body's axis, with x forward, y
    to the left and z up. The head frame has x forward and z up too; the camera sits 4 cm ahead of
    it and 5 cm above. With every joint at zero, the head frame is the base frame raised by
    0.1496 m. Pollen's SDK holds the head 0.177 m high by default, and gives its poses less that
    height.

    The rotation turns by `yaw` about z, then by `pitch` about the turned y, then by `roll` about
    the turned x: R = Rz(yaw) Ry(pitch) Rx(roll), as Pollen's SDK builds head poses.
    """

    x: float
    """In meters."""
    y: float
    """In meters."""
    z: float
    """In meters."""
    roll: float
    """In radians; positive tilts the head to its right."""
    pitch: float
    """In radians; positive looks down."""
    yaw: float
    """In radians; positive turns to the left."""
