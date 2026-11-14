"""The simulation's people, played: their clips one after the other, blinking, talking."""

from collections.abc import Sequence
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
    deform,
    displacement,
    load_clip,
    load_skin,
    pose,
    rest,
    retarget,
    sample,
)
from rva_robot_sim.speech import SpokenLine

STILL_BONES: Final = frozenset({"Bip01 REye", "Bip01 LEye", "Bip01 MJaw"})
"""Bones of the face the clips would move from their own rest pose, which is not the avatar's: the
eyes and the jaw keep the avatar's, and the face shapes move them."""
_BLINK_PERIOD_S: Final = 3.7
_BLINK_S: Final = 0.16


class Actor:
    """One person playing clips one after the other, over and over, from where they are placed.

    They blink, and their mouth opens as loud as their lines are while they say them. The clips'
    root motion carries over: a clip that turns them leaves the next facing the new way.
    """

    def __init__(
        self,
        skin: Skin,
        clips: Sequence[tuple[ClipName, Motion]],
        placement: Floats,
        lines: Sequence[SpokenLine],
    ) -> None:
        """Start the first clip.

        Args:
            skin: The avatar.
            clips: Retargeted to `skin`, each with its name; none, and they stand still at rest.
            placement: 4×4, where they stand on the floor at the start, from `people.standing`.
            lines: What they say, which the simulation plays too.
        """
        self.skin: Final = skin
        self._clips: Final = list(clips)
        self._lines: Final = list(lines)
        self._still: Final = rest(skin)
        self._blinks: Final = [
            skin.shape_index("AK_09_EyeBlinkLeft"),
            skin.shape_index("AK_10_EyeBlinkRight"),
        ]
        self._mouth: Final = skin.shape_index("AA_VI_10_aa")
        self._weights: Final = np.zeros(len(skin.shape_names))
        self._index = 0
        self._started = 0.0
        self.anchor = placement
        """4×4: where the current clip started on the floor."""

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
        motion = self._advance(time)
        rotations, root = sample(motion, time - self._started)
        return pose(self.skin, rotations, root, self.anchor)

    def vertices(self, bones: Floats, time: float) -> Floats:
        """V×3: the skin's vertices posed by `bones`, the face as it is at `time`."""
        self._weights[self._blinks] = _blink(time)
        self._weights[self._mouth] = max((line.opening(time) for line in self._lines), default=0.0)
        return deform(self.skin, bones, self._weights)

    def _advance(self, time: float) -> Motion:
        if not self._clips:
            return self._still
        while True:
            motion = self._clips[self._index][1]
            if time - self._started < motion.duration:
                return motion
            self.anchor = self.anchor @ displacement(self.skin, motion, motion.duration)
            self._started += motion.duration
            self._index = (self._index + 1) % len(self._clips)
            upcoming = self._clips[self._index][0]
            logger.debug("Playing {}/{}", upcoming.motion, upcoming.name)


def cast(
    config: AvatarsConfig, person: Person, lines: Sequence[SpokenLine], client: httpx.Client
) -> tuple[Path, Actor]:
    """The person's avatar, to load in Genesis, and the actor that plays its clips and lines.

    The avatar and the clips are downloaded and baked with Blender on first use.

    Raises:
        RuntimeError: If Blender fails to bake one.
        httpx.HTTPError: If a download fails, e.g. for an avatar or a clip that does not exist.
    """
    glb, skin_path = install_skin(config, person.avatar, client)
    skin = load_skin(skin_path)
    clips = [
        (name, retarget(skin, load_clip(install_clip(config, name, client)), STILL_BONES))
        for name in person.clips
    ]
    return glb, Actor(skin, clips, standing(person.position, person.turned_deg), lines)


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
