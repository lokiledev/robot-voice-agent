"""The simulation's people, played: idling, acting their scripts at their times, blinking."""

import functools
import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, assert_never

import httpx
import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import Field

from rva_pydantic import RVABaseModel
from rva_robot_sim.animation_assets import ClipName, install_clip, install_skin
from rva_robot_sim.people import AvatarsConfig, standing
from rva_robot_sim.scenarios import Play, Say, Script
from rva_robot_sim.scenes import Person
from rva_robot_sim.skinning import (
    Floats,
    Motion,
    Skin,
    deform,
    displacement,
    load_clip,
    load_skin,
    pose,
    rest,
    retarget,
    sample,
    truncate,
)
from rva_robot_sim.speech import SpokenLine

STILL_BONES: Final = frozenset({"Bip01 REye", "Bip01 LEye", "Bip01 MJaw"})
"""Bones of the face the clips would move from their own rest pose, which is not the avatar's: the
eyes and the jaw keep the avatar's, and the face shapes move them."""
HEAD_BONE: Final = "Bip01 Head"
"""The bone a person's voice comes from, near enough."""
_BLINK_PERIOD_S: Final = 3.7
_BLINK_S: Final = 0.16


class Action(RVABaseModel, arbitrary_types_allowed=True):
    """Clips played one after the other from a time, cutting the idle clip playing then."""

    at_s: float
    """When it starts, in seconds of simulated time."""
    clips: tuple[tuple[ClipName, Motion], ...] = Field(min_length=1)

    @property
    def end_s(self) -> float:
        return self.at_s + sum(motion.duration for _, motion in self.clips)


class Actor:
    """One person playing their idle clips over and over, and their actions at their times.

    They idle until their script begins, from when the actions' times count. An action cuts the
    idle clip playing when it starts; once it ends, the next idle clip starts. Their mouth stays
    shut until then too. They blink, and their mouth opens as their speech says: a script's or a live voice's.
    The clips' root motion carries over: a clip that turns them leaves the next facing the new
    way.
    """

    def __init__(
        self,
        skin: Skin,
        idle: Sequence[tuple[ClipName, Motion]],
        actions: Sequence[Action],
        placement: Floats,
        mouth: Callable[[float], float],
    ) -> None:
        """Start the first idle clip.

        Args:
            skin: The avatar.
            idle: Retargeted to `skin`, each with its name; none, and they stand still at rest
                between actions.
            actions: In order, retargeted to `skin`, at their times from the script's beginning.
            placement: 4×4, where they stand on the floor at the start, from `people.standing`.
            mouth: How open their mouth is at a time since the script began, from 0 to 1.

        Raises:
            ValueError: If an action starts before the one before it ends.
        """
        for before, after in itertools.pairwise(actions):
            if after.at_s < before.end_s:
                raise ValueError(
                    f"The step at {after.at_s:g} s starts before the one at {before.at_s:g} s"
                    f" ends, at {before.end_s:.2f} s"
                )
        self.skin: Final = skin
        self._idle: Final = list(idle)
        self._actions: Final = list(actions)
        self._opening: Final = mouth
        self._still: Final = rest(skin)
        self._blinks: Final = [
            skin.shape_index("AK_09_EyeBlinkLeft"),
            skin.shape_index("AK_10_EyeBlinkRight"),
        ]
        self._mouth: Final = skin.shape_index("AA_VI_10_aa")
        self._weights: Final = np.zeros(len(skin.shape_names))
        self._next_idle = 0
        self._next_action = 0
        self._begun: float | None = None
        # The clips of the action playing that are still to come; None while idle.
        self._acting: list[tuple[ClipName, Motion]] | None = None
        self._motion = self._idle_clip()
        self._started = 0.0
        self.anchor = placement
        """4×4: where the current clip started on the floor."""

    def begin(self, time: float) -> None:
        """Begin their script at `time`, in seconds: its actions count from then."""
        assert self._begun is None, "a script begins once"
        self._begun = time

    def place(self, placement: Floats) -> None:
        """Move them: the current clip goes on from `placement` as if it had started there."""
        self.anchor = placement

    def pose(self, time: float) -> Floats:
        """B×4×4: the bones in the world at `time`, in seconds, moving on to the next clips.

        Raises:
            ValueError: If `time` is before the last time asked.
        """
        if time < self._started:
            raise ValueError(f"Time went back from {self._started} to {time}")
        self._advance(time)
        rotations, root = sample(self._motion, time - self._started)
        return pose(self.skin, rotations, root, self.anchor)

    def vertices(self, bones: Floats, time: float) -> Floats:
        """V×3: the skin's vertices posed by `bones`, the face as it is at `time`."""
        self._weights[self._blinks] = _blink(time)
        self._weights[self._mouth] = self.voice(time)
        return deform(self.skin, bones, self._weights)

    def voice(self, time: float) -> float:
        """How loud they speak at `time`, from 0 to 1: how open their mouth is."""
        return 0.0 if self._begun is None else self._opening(time - self._begun)

    def head(self, bones: Floats) -> Floats:
        """3: where their head is, posed by `bones`, in the world."""
        return bones[self.skin.bone_names.index(HEAD_BONE), :3, 3]

    def _advance(self, time: float) -> None:
        while True:
            standing_still = self._acting is None and not self._idle
            ends = math.inf if standing_still else self._started + self._motion.duration
            cut = math.inf
            if (
                self._begun is not None
                and self._acting is None
                and self._next_action < len(self._actions)
            ):
                cut = self._begun + self._actions[self._next_action].at_s
            if time < min(ends, cut):
                return
            if cut <= ends:
                self._move_on(cut)
                self._acting = list(self._actions[self._next_action].clips)
                self._next_action += 1
                self._motion = self._action_clip(self._acting)
            else:
                self._move_on(ends)
                if self._acting is not None and len(self._acting) > 0:
                    self._motion = self._action_clip(self._acting)
                else:
                    self._acting = None
                    self._motion = self._idle_clip()

    def _move_on(self, time: float) -> None:
        """Leave the current clip at `time`: the next starts where it left them."""
        self.anchor = self.anchor @ displacement(self.skin, self._motion, time - self._started)
        self._started = time

    def _action_clip(self, acting: list[tuple[ClipName, Motion]]) -> Motion:
        name, motion = acting.pop(0)
        logger.debug("Acting {}/{}", name.motion, name.name)
        return motion

    def _idle_clip(self) -> Motion:
        if not self._idle:
            return self._still
        name, motion = self._idle[self._next_idle]
        self._next_idle = (self._next_idle + 1) % len(self._idle)
        logger.debug("Idling {}/{}", name.motion, name.name)
        return motion


def cast(
    config: AvatarsConfig,
    person: Person,
    script: Script,
    lines: Sequence[SpokenLine],
    mouth: Callable[[float], float],
    client: httpx.Client,
) -> tuple[Path, Actor]:
    """The person's avatar, to load in Genesis, and the actor that plays its script.

    A line's gesture lasts as long as the line. The avatar and the clips are downloaded and
    baked with Blender on first use.

    Args:
        config: Where the avatars and the clips come from.
        person: Who they are.
        script: What they do.
        lines: What they say, the script's `Say` steps spoken, in order.
        mouth: How open their mouth is at a time, from 0 to 1.
        client: For the downloads.

    Raises:
        RuntimeError: If Blender fails to bake one.
        httpx.HTTPError: If a download fails, e.g. for an avatar or a clip that does not exist.
        ValueError: If a step starts before the one before it ends, as long as its clips and
            lines turn out.
    """
    glb, skin_path = install_skin(config, person.avatar, client)
    skin = load_skin(skin_path)

    @functools.cache
    def motion(name: ClipName) -> Motion:
        return retarget(skin, load_clip(install_clip(config, name, client)), STILL_BONES)

    spoken = iter(lines)
    actions = []
    for step in script.steps:
        match step:
            case Say(gesture=gesture):
                line = next(spoken)
                actions.append(saying(step.at_s, line.clip.duration, gesture, motion(gesture)))
            case Play(clips=clips):
                played = tuple((clip, motion(clip)) for clip in clips)
                actions.append(Action(at_s=step.at_s, clips=played))
            case _:
                assert_never(step)
    assert next(spoken, None) is None, "one line per Say step"
    idle = [(name, motion(name)) for name in script.idle]
    try:
        actor = Actor(skin, idle, actions, standing(person.position, person.turned_deg), mouth)
    except ValueError as error:
        raise ValueError(f"{person.name}: {error}") from error
    return glb, actor


def saying(at_s: float, line_s: float, name: ClipName, gesture: Motion) -> Action:
    """Gesturing while saying a line `line_s` seconds long from `at_s`, as long as it lasts.

    The gesture repeats while the line goes on, and is cut where it ends: Rocketbox's talking
    clips are long takes, half a minute, not one gesture each.
    """
    whole, rest = divmod(line_s, gesture.duration)
    clips = [(name, gesture)] * int(whole)
    if rest > 0:
        clips.append((name, truncate(gesture, rest)))
    return Action(at_s=at_s, clips=tuple(clips))


def vertex_order(skin: Skin, loaded: Floats) -> npt.NDArray[np.intp]:
    """For each vertex Genesis loaded, the skin's vertex at the same place.

    Genesis splits vertices where the texture's seams are, and may reorder them: copies of a vertex
    share its rest position, and so its bones and weights.

    Raises:
        ValueError: If a loaded vertex is not one of the skin's.
    """
    nearest = np.empty(len(loaded), dtype=np.intp)
    distance = np.empty(len(loaded))
    for start in range(0, len(loaded), 1024):
        chunk = loaded[start : start + 1024]
        squared = ((chunk[:, None, :] - skin.positions[None, :, :]) ** 2).sum(axis=-1)
        nearest[start : start + 1024] = squared.argmin(axis=1)
        distance[start : start + 1024] = np.sqrt(squared.min(axis=1))
    if distance.max() > 1e-4:
        raise ValueError(f"A vertex Genesis loaded is {distance.max():.3g} m from the skin's")
    return nearest


def _blink(time: float) -> float:
    phase = time % _BLINK_PERIOD_S
    return max(0.0, 1.0 - abs(phase - _BLINK_S / 2) / (_BLINK_S / 2))
