"""The simulation's people, played: idling, acting what they are told to, blinking."""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import numpy as np
import numpy.typing as npt
from loguru import logger

from rva_robot_sim.animation_assets import ClipName, install_clip, install_skin
from rva_robot_sim.people import AvatarsConfig, standing
from rva_robot_sim.scenes import Person
from rva_robot_sim.skinning import (
    Floats,
    Motion,
    Skin,
    aimed,
    deform,
    displacement,
    heading,
    load_clip,
    load_skin,
    placement,
    pose,
    rest,
    retarget,
    sample,
    truncate,
)

STILL_BONES: Final = frozenset({"Bip01 REye", "Bip01 LEye", "Bip01 MJaw"})
"""Bones of the face the clips would move from their own rest pose, which is not the avatar's: the
eyes and the jaw keep the avatar's, and the face shapes move them."""
HEAD_BONE: Final = "Bip01 Head"
"""The bone a person's voice comes from, near enough."""
_CARRYING_HAND: Final = "Bip01 L Hand"
_CARRYING_ARM: Final = (
    # The upper arm down along the body, the forearm level ahead: -y is ahead, +x their left.
    ("Bip01 L UpperArm", "Bip01 L Forearm", (0.15, 0.0, -1.0)),
    ("Bip01 L Forearm", _CARRYING_HAND, (-0.25, -1.0, 0.05)),
)
_CARRIED_FROM_HAND: Final = (-0.02, -0.05, 0.03)
"""Where the middle of what they carry rests, from the hand under it: to their left, behind and
up, in meters."""
_BLINK_PERIOD_S: Final = 3.7
_BLINK_S: Final = 0.16


class Actor:
    """One person playing their idle clips over and over, and what they are told to in between.

    Playing clips or gesturing cuts the clip playing; once they end, the next idle clip starts.
    They blink, and their mouth opens as their speech says: a line's or a live voice's. The clips'
    root motion carries over: a clip that turns them leaves the next facing the new way.
    """

    def __init__(
        self,
        skin: Skin,
        idle: Sequence[ClipName],
        motions: Mapping[ClipName, Motion],
        placement: Floats,
        mouth: Callable[[float], float],
    ) -> None:
        """Start the first idle clip.

        Args:
            skin: The avatar.
            idle: Played between what they are told to; none, and they stand still at rest.
            motions: Every clip they may play, the idle ones too, retargeted to `skin`.
            placement: 4×4, where they stand on the floor at the start, from `people.standing`.
            mouth: How open their mouth is at a time, in seconds, from 0 to 1.
        """
        self.skin: Final = skin
        self._idle: Final = list(idle)
        self._motions: Final = motions
        self._opening: Final = mouth
        self._still: Final = rest(skin)
        self._blinks: Final = [
            skin.shape_index("AK_09_EyeBlinkLeft"),
            skin.shape_index("AK_10_EyeBlinkRight"),
        ]
        self._mouth: Final = skin.shape_index("AA_VI_10_aa")
        self._weights: Final = np.zeros(len(skin.shape_names))
        self._next_idle = 0
        # The clips still to come of what they were told to play; None while idle.
        self._acting: list[tuple[ClipName, Motion]] | None = None
        self._motion = self._idle_clip()
        self._started = 0.0
        self.anchor = placement
        """4×4: where the current clip started on the floor."""

    def play(self, clips: Sequence[ClipName], time: float) -> float:
        """Play `clips` once from `time`, one after the other, cutting what plays then.

        Returns:
            How long they last together, in seconds.
        """
        acting = [(name, self._motions[name]) for name in clips]
        lasts = sum(motion.duration for _, motion in acting)
        self._act(acting, time)
        return lasts

    def gesture(self, name: ClipName, seconds: float, time: float) -> None:
        """Gesture with the clip `name` from `time`, for `seconds`: as long as a line lasts."""
        self._act(lasting(name, self._motions[name], seconds), time)

    def cut(self, time: float) -> None:
        """Stop what they play at `time`: they idle from where they are."""
        self._advance(time)
        self._move_on(time)
        self._acting = None
        self._motion = self._idle_clip()

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
        return self._opening(time)

    def head(self, bones: Floats) -> Floats:
        """3: where their head is, posed by `bones`, in the world."""
        return bones[self.skin.bone_names.index(HEAD_BONE), :3, 3]

    def carrying(self, bones: Floats) -> Floats:
        """4×4: where the bottom of what they carry is, posed by `bones`: level, turned as they."""
        turned = bones[0, :3, :3] @ self.skin.bone_rest[0, :3, :3].T
        frame = placement(0.0, 0.0, heading(turned))
        hand = bones[self.skin.bone_names.index(_CARRYING_HAND), :3, 3]
        frame[:3, 3] = hand + frame[:3, :3] @ _CARRIED_FROM_HAND
        return frame

    def _act(self, acting: list[tuple[ClipName, Motion]], time: float) -> None:
        self._advance(time)
        self._move_on(time)
        self._acting = acting
        self._motion = self._action_clip(acting)

    def _advance(self, time: float) -> None:
        while True:
            standing_still = self._acting is None and not self._idle
            ends = math.inf if standing_still else self._started + self._motion.duration
            if time < ends:
                return
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
        name = self._idle[self._next_idle]
        motion = self._motions[name]
        self._next_idle = (self._next_idle + 1) % len(self._idle)
        logger.debug("Idling {}/{}", name.motion, name.name)
        return motion


def cast(
    config: AvatarsConfig,
    person: Person,
    idle: Sequence[ClipName],
    clips: Iterable[ClipName],
    mouth: Callable[[float], float],
    client: httpx.Client,
) -> tuple[Path, Actor]:
    """The person's avatar, to load in Genesis, and the actor that plays them.

    The avatar and the clips are downloaded and baked with Blender on first use.

    Args:
        config: Where the avatars and the clips come from.
        person: Who they are.
        idle: What they play between what they are told to.
        clips: Every other clip they may be told to play or gesture with.
        mouth: How open their mouth is at a time, from 0 to 1.
        client: For the downloads.

    Raises:
        RuntimeError: If Blender fails to bake one.
        httpx.HTTPError: If a download fails, e.g. for an avatar or a clip that does not exist.
    """
    glb, skin_path = install_skin(config, person.avatar, client)
    skin = load_skin(skin_path)
    motions = {
        name: retarget(skin, load_clip(install_clip(config, name, client)), STILL_BONES)
        for name in dict.fromkeys((*idle, *clips))
    }
    if person.carries is not None:
        # The clips swing both arms: the one under what they carry holds still instead.
        motions = {name: aimed(skin, motion, _CARRYING_ARM) for name, motion in motions.items()}
    return glb, Actor(skin, idle, motions, standing(person.position, person.turned_deg), mouth)


def lasting(name: ClipName, gesture: Motion, seconds: float) -> list[tuple[ClipName, Motion]]:
    """The clips of a gesture that lasts `seconds`, as a line does.

    The gesture repeats while the line goes on, and is cut where it ends: Rocketbox's talking
    clips are long takes, half a minute, not one gesture each.
    """
    whole, rest = divmod(seconds, gesture.duration)
    clips = [(name, gesture)] * int(whole)
    if rest > 0:
        clips.append((name, truncate(gesture, rest)))
    return clips


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
