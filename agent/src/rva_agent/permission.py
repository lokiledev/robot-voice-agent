from collections.abc import Generator
from contextlib import contextmanager
from typing import Final
from uuid import UUID

from rva_schemas.abc.vision import Vision
from rva_schemas.profile import Admins
from rva_schemas.vision import KnownPerson


class ActionPermission:
    """Who may make the robot act, and authorize or revoke others: operators.

    Operators are the admins, listed in a file of the machine, and the people an operator
    authorized. They are only obeyed while the camera recognizes them as the nearest face, but
    what they started runs to its end: the robot may go on with it once they are out of view.
    Without a camera nobody can be recognized, so anyone may make the robot act.
    """

    def __init__(self, vision: Vision | None, admins: Admins) -> None:
        self._vision: Final = vision
        self._admins: Final = admins
        self._going_on = False

    def is_admin(self, user_id: UUID) -> bool:
        return user_id in self._admins.user_ids

    def is_operator(self, user_id: UUID) -> bool:
        return self.is_admin(user_id) or (
            self._vision is not None and user_id in self._vision.granted_operators()
        )

    def operators(self) -> frozenset[UUID]:
        granted = frozenset[UUID]() if self._vision is None else self._vision.granted_operators()
        return self._admins.user_ids | granted

    @contextmanager
    def going_on(self) -> Generator[None]:
        """While the robot answers the end of an action that an operator had it start.

        Its next step is theirs too, wherever they are by now: an operator starts a plan, and
        need not stay in view until its last step.
        """
        self._going_on = True
        try:
            yield
        finally:
            self._going_on = False

    def refusal(self) -> str | None:
        """Why the robot may not act now, or None if it may.

        It may for the operator in front of it, and to go on with what an operator started.
        """
        if self._vision is None or self._going_on:
            return None
        requester = self.operator_in_front()
        return requester if isinstance(requester, str) else None

    def operator_in_front(self) -> KnownPerson | str:
        """The operator in front of the robot, or why the person there is not one."""
        if self._vision is None:
            return "Without a camera you cannot tell who is in front of you."
        nearest = next((face for face in self._vision.faces() if face.nearest), None)
        if nearest is None:
            return "Nobody is in front of you."
        if nearest.person is None:
            return "The person in front of you is not one of your operators."
        if not self.is_operator(nearest.person.user_id):
            return f"{nearest.person.name} is not one of your operators."
        return nearest.person
