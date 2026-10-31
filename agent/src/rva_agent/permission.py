from typing import Final

from rva_schemas.abc.vision import Vision
from rva_schemas.profile import Operators


class ActionPermission:
    """Whether the person in front of the robot may make it act.

    Only operators may, once the camera recognizes them as the nearest face. Without a camera
    nobody can be recognized, so anyone may.
    """

    def __init__(self, vision: Vision | None, operators: Operators) -> None:
        self._vision: Final = vision
        self._operators: Final = operators

    def refusal(self) -> str | None:
        """Why the person in front of the robot may not make it act, or None if they may."""
        if self._vision is None:
            return None
        nearest = next((face for face in self._vision.faces() if face.nearest), None)
        if nearest is None:
            return "Nobody is in front of you, so nobody can ask you to move."
        if nearest.person is None:
            return "The person in front of you is not one of your operators, so they cannot ask you to move."
        if nearest.person.user_id not in self._operators.user_ids:
            return f"{nearest.person.name} is not one of your operators, so they cannot ask you to move."
        return None
