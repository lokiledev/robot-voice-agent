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
    skills: tuple[Skill, ...]
    constraints: tuple[str, ...]
    """Things the robot cannot do, e.g. "cannot climb stairs"."""


class Operators(RVABaseModel):
    """People allowed to make the robot act, by the user id of their face.

    Only ids are listed, so the profile holds no names. It lives in the profile so the robot can
    never add an operator itself: anyone can tell it a name.
    """

    user_ids: frozenset[UUID]


class Profile(RVABaseModel):
    """Who the robot is and where it is deployed, loaded from a profile directory."""

    character: Character
    environment: Environment
    capabilities: Capabilities
    operators: Operators

    @classmethod
    def load(cls, directory: Path) -> "Profile":
        return cls(
            character=load_yaml(Character, directory / "character.yaml"),
            environment=load_yaml(Environment, directory / "environment.yaml"),
            capabilities=load_yaml(Capabilities, directory / "capabilities.yaml"),
            operators=load_yaml(Operators, directory / "operators.yaml"),
        )
