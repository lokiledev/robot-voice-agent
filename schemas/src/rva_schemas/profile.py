from pathlib import Path
from typing import Literal
from uuid import UUID

from rva_pydantic import RVABaseModel, load_yaml


class Character(RVABaseModel):
    name: str
    personality: str
    speaking_style: str
    verbosity: Literal["terse", "normal", "chatty"]


class Place(RVABaseModel):
    name: str
    description: str


class KnownObject(RVABaseModel):
    name: str
    description: str
    usual_place: str | None
    """Name of the place where it is usually found, if it has one."""


class Person(RVABaseModel):
    name: str
    description: str


class Environment(RVABaseModel):
    location: str
    description: str
    places: tuple[Place, ...]
    objects: tuple[KnownObject, ...]
    people: tuple[Person, ...]


class Skill(RVABaseModel):
    name: str
    description: str
    examples: tuple[str, ...]
    """Commands phrased the way the VLA was trained on them."""


class Capabilities(RVABaseModel):
    body: str
    """What the robot is, as it introduces itself, e.g. "a humanoid robot"."""
    skills: tuple[Skill, ...]
    """What VLA tasks can do; none for a robot without a VLA, which then has no task tools."""
    walks: bool
    """Whether it can walk to places and to people."""
    animations: tuple[str, ...]
    """Body animations that can be played, e.g. "wave_right_hand"."""
    constraints: tuple[str, ...]
    """Things the robot cannot do, e.g. "cannot climb stairs"."""


class Admins(RVABaseModel):
    """Operators who cannot be revoked by voice, by the user id of their face.

    They are listed in a file of the machine, outside the profile and the source tree, since the ids
    refer to its face database; only ids, so it holds no names. The robot can never make someone an
    admin: other operators are granted by an operator, by voice.
    """

    user_ids: frozenset[UUID]


class Profile(RVABaseModel):
    """Who the robot is and where it is deployed, loaded from a profile directory."""

    character: Character
    environment: Environment
    capabilities: Capabilities

    @classmethod
    def load(cls, directory: Path) -> "Profile":
        return cls(
            character=load_yaml(Character, directory / "character.yaml"),
            environment=load_yaml(Environment, directory / "environment.yaml"),
            capabilities=load_yaml(Capabilities, directory / "capabilities.yaml"),
        )
