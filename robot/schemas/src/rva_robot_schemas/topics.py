"""The robot's kickmsg topics, in the namespace both ends are configured with.

The robot, simulated or real, creates them all when it starts; clients open them and wait for it
if it is not up yet. Joint states and commands travel as the JSON of their model, frames as
`encode_frame` writes them.
"""

from typing import Final

ROBOT_NODE: Final = "robot"
"""The robot's node, which owns the command mailbox."""
JOINT_STATE: Final = "joint_state"
"""`JointState`, published by the robot."""
CAMERA: Final = "camera"
"""`CameraFrame` from the head camera, published by the robot."""
JOINT_COMMAND: Final = "joint_command"
"""`JointCommand`, a mailbox of the robot's node that any client may send to."""
