import json
from collections.abc import AsyncIterator, Callable
from typing import Final
from uuid import UUID

import pydantic
from loguru import logger
from pydantic import Field

from rva_agent.permission import ActionPermission
from rva_agent.tasks import TaskSupervisor
from rva_agent.tools import ToolResult
from rva_body_schemas.abc.client import BodyClient
from rva_body_schemas.errors import BodyBusyError
from rva_body_schemas.motion import (
    GoToHuman,
    GoToLocation,
    Motion,
    MotionEvent,
    MotionRequest,
    PlayAnimation,
    describe,
)
from rva_pydantic import RVABaseModel
from rva_schemas.abc.vision import Vision
from rva_schemas.dialogue import ToolCall, ToolSpec
from rva_vla_schemas.task import FinalTaskStatus, TaskRunning


class BodySupervisor:
    """Keeps track of the one motion the body is making: walking or an animation."""

    def __init__(self, body: BodyClient, on_end: Callable[[Motion], None]) -> None:
        """Track motions of `body`.

        Args:
            body: The robot's body.
            on_end: Called with the final state of each motion started here.
        """
        self._body: Final = body
        self._on_end: Final = on_end
        self._active: Motion | None = None
        # A short motion can end before `start` returns; its end is held here until then.
        self._early_ends: Final[dict[UUID, FinalTaskStatus]] = {}

    @property
    def active(self) -> Motion | None:
        return self._active

    @property
    def body(self) -> BodyClient:
        return self._body

    async def watch(self, events: AsyncIterator[MotionEvent]) -> None:
        """Follow motion events until `events` ends. Run it for the whole session."""
        async for event in events:
            match event.status:
                case TaskRunning():
                    pass
                case status:
                    self._end(event.motion_id, status)

    async def start(self, action: GoToLocation | GoToHuman | PlayAnimation) -> Motion:
        """Start a motion and make it the active one.

        Raises:
            BodyBusyError: If the body is already moving or running a VLA task.
        """
        motion = await self._body.start(MotionRequest(action=action))
        self._active = motion
        if (status := self._early_ends.pop(motion.id, None)) is not None:
            self._end(motion.id, status)
        return motion

    async def stay(self) -> None:
        """Stop the running motion, if any; its end is reported through `on_end`."""
        await self._body.stay()

    def _end(self, motion_id: UUID, status: FinalTaskStatus) -> None:
        if self._active is None or self._active.id != motion_id:
            self._early_ends[motion_id] = status
            return
        motion = Motion(id=motion_id, action=self._active.action, status=status)
        self._active = None
        logger.info("Motion {!r} ended: {}", describe(motion.action), status.state)
        self._on_end(motion)


class GoToLocationArgs(RVABaseModel):
    location: str = Field(min_length=1)
    """One of the places of the environment."""


class UserArgs(RVABaseModel):
    user_id: UUID
    """User id of a person in view, from who_is_here."""


class PlayAnimationArgs(RVABaseModel):
    animation: str = Field(min_length=1)
    """One of the animations listed in your instructions."""


class NoArgs(RVABaseModel):
    pass


_MOVING: Final = (
    ToolSpec(
        name="go_to_location",
        description="Walk to one of the places of the environment. Returns once you start walking; you are told when you arrive.",
        parameters=GoToLocationArgs.model_json_schema(),
    ),
    ToolSpec(
        name="stay",
        description="Stop walking or gesturing and stay where you are.",
        parameters=NoArgs.model_json_schema(),
    ),
    ToolSpec(
        name="play_animation",
        description="Play a body animation, e.g. a wave or a dance.",
        parameters=PlayAnimationArgs.model_json_schema(),
    ),
)
_WITH_PEOPLE: Final = (
    ToolSpec(
        name="go_to_human",
        description="Walk to a person in view, by their user_id from who_is_here.",
        parameters=UserArgs.model_json_schema(),
    ),
    ToolSpec(
        name="look_at_user",
        description="Keep your head and body turned towards a person in view, without walking, by their user_id from who_is_here.",
        parameters=UserArgs.model_json_schema(),
    ),
    ToolSpec(
        name="stop_look_at_user",
        description="Stop following someone with your head and body.",
        parameters=NoArgs.model_json_schema(),
    ),
)


class BodyTools:
    """The tools the LLM uses to walk, gesture and look at people, besides VLA tasks.

    Walking needs an operator in front of the robot, like tasks; gestures, looking and staying
    do not. Walking, animations and tasks share the body, so only one runs at a time.
    """

    names: Final = frozenset(spec.name for spec in (*_MOVING, *_WITH_PEOPLE))

    def __init__(
        self,
        body: BodySupervisor,
        tasks: TaskSupervisor,
        permission: ActionPermission,
        vision: Vision | None,
        locations: tuple[str, ...],
        animations: tuple[str, ...],
    ) -> None:
        self._body: Final = body
        self._tasks: Final = tasks
        self._permission: Final = permission
        self._vision: Final = vision
        self._locations: Final = locations
        self._animations: Final = animations
        # Walking to or looking at someone needs to see them.
        self.specs: Final = _MOVING + (() if vision is None else _WITH_PEOPLE)

    async def call(self, call: ToolCall) -> ToolResult:
        """Run a tool call and return its result for the LLM; its mistakes are results too."""
        try:
            match call.name:
                case "go_to_location":
                    return await self._go_to_location(
                        GoToLocationArgs.model_validate_json(call.arguments)
                    )
                case "go_to_human":
                    return await self._go_to_human(UserArgs.model_validate_json(call.arguments))
                case "stay":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    await self._body.stay()
                    return _done(status="staying where you are")
                case "play_animation":
                    return await self._play(PlayAnimationArgs.model_validate_json(call.arguments))
                case "look_at_user":
                    return await self._look_at(UserArgs.model_validate_json(call.arguments))
                case "stop_look_at_user":
                    NoArgs.model_validate_json(call.arguments or "{}")
                    await self._body.body.stop_looking()
                    return _done(status="no longer following anyone with your gaze")
                case _:
                    return _error(f"There is no tool named {call.name}.")
        except pydantic.ValidationError as error:
            return _error(f"Invalid arguments: {error}")

    async def _go_to_location(self, args: GoToLocationArgs) -> ToolResult:
        if (refusal := self._permission.refusal()) is not None:
            return _error(f"{refusal} Only operators can ask you to walk somewhere.")
        location = next(
            (
                place
                for place in self._locations
                if place.casefold() == args.location.strip().casefold()
            ),
            None,
        )
        if location is None:
            return _error(
                f"{args.location} is not a place you know. Places: {', '.join(self._locations)}."
            )
        return await self._start(GoToLocation(location=location), None)

    async def _go_to_human(self, args: UserArgs) -> ToolResult:
        if (refusal := self._permission.refusal()) is not None:
            return _error(f"{refusal} Only operators can ask you to walk somewhere.")
        if (name := self._name_in_view(args.user_id)) is None:
            return _error("Nobody with this user_id is in view. Call who_is_here.")
        return await self._start(GoToHuman(user_id=args.user_id), name)

    async def _play(self, args: PlayAnimationArgs) -> ToolResult:
        if args.animation not in self._animations:
            return _error(
                f"There is no animation {args.animation}. Animations: {', '.join(self._animations)}."
            )
        return await self._start(PlayAnimation(animation=args.animation), None)

    async def _look_at(self, args: UserArgs) -> ToolResult:
        if (name := self._name_in_view(args.user_id)) is None:
            return _error("Nobody with this user_id is in view. Call who_is_here.")
        await self._body.body.look_at(args.user_id)
        return _done(status=f"looking at {name}")

    async def _start(
        self, action: GoToLocation | GoToHuman | PlayAnimation, target: str | None
    ) -> ToolResult:
        """Start a motion; `target` names the person walked to, if any."""
        if (task := self._tasks.active) is not None:
            return _error(f"Your body is busy with {task.instruction!r}. Cancel it first or wait.")
        try:
            await self._body.start(action)
        except BodyBusyError as error:
            return _error(f"Your body is busy: {error.busy_with}. Call stay first or wait.")
        what = describe(action) if target is None else f"go to {target}"
        return _done(status="started", motion=what, next="You are told when it ends.")

    def _name_in_view(self, user_id: UUID) -> str | None:
        assert self._vision is not None, "people tools are only offered with a camera"
        return next(
            (
                face.person.name
                for face in self._vision.faces()
                if face.person is not None and face.person.user_id == user_id
            ),
            None,
        )


def _done(**fields: str) -> ToolResult:
    # Like a task start, nothing to add when the LLM announced it.
    return ToolResult(content=json.dumps(fields), needs_reply=False)


def _error(message: str) -> ToolResult:
    return ToolResult(content=json.dumps({"error": message}), needs_reply=True)
