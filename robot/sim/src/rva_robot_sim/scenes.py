"""What the simulated Reachy Mini sees around it: the people, where they stand and what they do.

Scenes are written as code: each is a function that describes one, picked by its name in
`SCENES`, e.g. `lune-sim --scene animated_pair`.
"""

from collections.abc import Callable, Mapping
from typing import Final

from rva_pydantic import RVABaseModel
from rva_robot_sim.animation_assets import ClipName


class Person(RVABaseModel):
    """One of a scene's people, at the start."""

    avatar: str
    """A Rocketbox avatar, by its directory under `Assets/Avatars`, e.g. `Adults/Female_Adult_01`
    to `17`, `Adults/Male_Adult_01` to `21`."""
    position: tuple[float, float]
    """Where they stand on the floor, in meters, in the robot's frame: it faces +x, +y is its
    left."""
    turned_deg: float
    """How far they are turned from facing the robot, in degrees, counterclockwise seen from
    above; 0 faces it."""
    clips: tuple[ClipName, ...]
    """Rocketbox clips they play one after the other, over and over, blinking, their mouth
    moving in the talking ones. None, and they stand still in their A-pose. A clip that turns
    them leaves the next one facing the new way."""


class SceneDescription(RVABaseModel):
    people: tuple[Person, ...]
    """In order: `npc` commands and the `person_pose` mailbox name them by their index here."""


def still_pair() -> SceneDescription:
    """Two people standing still in an A-pose, facing the robot."""
    return SceneDescription(people=(_in_front(()), _on_the_right(())))


def animated_pair() -> SceneDescription:
    """The same two, alive: she talks to the robot with gestures, he listens.

    Both idle in between, and look around now and then.
    """
    return SceneDescription(
        people=(
            _in_front(
                _clips(
                    "static/f_idle_neutral_01",
                    "static/f_gestic_talk_neutral_01",
                    "static/f_idle_look_around_01",
                    "static/f_gestic_talk_neutral_02",
                )
            ),
            _on_the_right(
                _clips(
                    "static/m_gestic_listen_neutral_01",
                    "static/m_idle_neutral_01",
                    "static/m_idle_look_around_01",
                    "static/m_gestic_listen_neutral_02",
                )
            ),
        )
    )


SCENES: Final[Mapping[str, Callable[[], SceneDescription]]] = {
    "still_pair": still_pair,
    "animated_pair": animated_pair,
}
"""Every scene, by the name `--scene` takes."""


def describe(name: str) -> SceneDescription:
    """The scene `name` describes.

    Raises:
        KeyError: If there is no such scene; its message lists them.
    """
    if name not in SCENES:
        raise KeyError(f"No scene {name!r}; there are {sorted(SCENES)}")
    return SCENES[name]()


def _in_front(clips: tuple[ClipName, ...]) -> Person:
    """1.1 m in front of the robot, facing it."""
    return Person(avatar="Adults/Female_Adult_01", position=(1.1, 0.0), turned_deg=0.0, clips=clips)


def _on_the_right(clips: tuple[ClipName, ...]) -> Person:
    """On the robot's right, 40 degrees off its axis and a little closer, facing it too.

    The nearest face: the one `lune see` names with n while the other is already known.
    """
    return Person(avatar="Adults/Male_Adult_04", position=(0.8, -0.65), turned_deg=0.0, clips=clips)


def _clips(*names: str) -> tuple[ClipName, ...]:
    return tuple(ClipName.parse(name) for name in names)
