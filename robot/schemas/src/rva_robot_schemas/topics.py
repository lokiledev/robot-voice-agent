"""The robot's kickmsg topics, in the namespace both ends are configured with.

The robot, simulated or real, creates its topics when it starts, and the head controller its
mailboxes; clients open them and wait for them if they are not up yet. Joint states, commands,
head poses and gaze targets travel as the JSON of their model, frames as `encode_frame` writes
them.
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
HEAD_CONTROLLER_NODE: Final = "head_controller"
"""The head controller's node, which owns the head and gaze target mailboxes."""
HEAD_TARGET: Final = "head_target"
"""`HeadPose`, a mailbox of the head controller's node: where to put the head."""
GAZE_TARGET: Final = "gaze_target"
"""`GazeTarget`, a mailbox of the head controller's node: what the head camera should look at."""
