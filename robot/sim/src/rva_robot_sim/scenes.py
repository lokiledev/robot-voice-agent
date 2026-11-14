"""What the simulated Reachy Mini has around it at the start: its table and the people.

Scenes are written as code, each a function; a scenario (`scenarios.py`) picks one and says what
happens in it.
"""

from pydantic import field_validator

from rva_pydantic import RVABaseModel


class Person(RVABaseModel):
    """One of a scene's people: who they are and where they stand at the start."""

    name: str
    """How scenarios, `npc` commands and the `person_pose` mailbox call them, e.g. `mia`."""
    avatar: str
    """A Rocketbox avatar, by its directory under `Assets/Avatars`, e.g. `Adults/Female_Adult_01`
    to `17`, `Adults/Male_Adult_01` to `21`."""
    voice: str
    """Kokoro's voice for what they say, e.g. `af_bella` or `am_michael`."""
    position: tuple[float, float]
    """Where they stand on the floor, in meters, in the robot's frame: it faces +x, +y is its
    left."""
    turned_deg: float
    """How far they are turned from facing the robot, in degrees, counterclockwise seen from
    above; 0 faces it."""


class Scene(RVABaseModel):
    table_height: float
    """Of the table top the robot stands on, in meters. The head camera sits about 0.15 m above
    it."""
    people: tuple[Person, ...]

    @field_validator("people")
    @classmethod
    def _named_once(cls, people: tuple[Person, ...]) -> tuple[Person, ...]:
        names = [person.name for person in people]
        if len(set(names)) != len(names):
            raise ValueError(f"People named {names}: each name once")
        return people

    def person(self, name: str) -> Person:
        """The person called `name`.

        Raises:
            KeyError: If nobody has this name; its message names those who do.
        """
        for person in self.people:
            if person.name == name:
                return person
        raise KeyError(f"Nobody named {name!r}; there are {[p.name for p in self.people]}")


def two_by_the_table() -> Scene:
    """Mia 1.1 m in front of the robot, and Ben on its right, both facing it.

    Ben stands 40 degrees off its axis and a little closer: the nearest face, the one `lune see`
    names with n while the other is already known. The table puts the head camera about 1.25 m
    above the floor: a seated person's face, a little below a standing one's.
    """
    return Scene(
        table_height=1.1,
        people=(
            Person(
                name="mia",
                avatar="Adults/Female_Adult_01",
                voice="af_bella",
                position=(1.1, 0.0),
                turned_deg=0.0,
            ),
            Person(
                name="ben",
                avatar="Adults/Male_Adult_04",
                voice="am_michael",
                position=(0.8, -0.65),
                turned_deg=0.0,
            ),
        ),
    )


def chatting_by_the_table() -> Scene:
    """Mia and Ben chatting in front of the robot, and Cleo out of its view on its right.

    Mia and Ben stand 0.6 m apart, each turned 60 degrees to the other: their faces are seen, but
    not looking at the robot. A 60-degree turn clip faces each of them to it. Cleo stands 3.6 m
    to the robot's right, 74 degrees off its axis and out of view, facing its left: Rocketbox's
    walk clips take her to (1.06, -0.75), 35 degrees off its axis, where a 60-degree turn to her
    left faces her to it.
    """
    return Scene(
        table_height=1.1,
        people=(
            Person(
                name="mia",
                avatar="Adults/Female_Adult_01",
                voice="af_bella",
                position=(1.2, 0.45),
                turned_deg=60.0,
            ),
            Person(
                name="ben",
                avatar="Adults/Male_Adult_04",
                voice="am_michael",
                position=(1.2, -0.15),
                turned_deg=-60.0,
            ),
            Person(
                name="cleo",
                avatar="Adults/Female_Adult_05",
                voice="bf_emma",
                position=(1.0, -3.6),
                # Facing the robot's left, along +y.
                turned_deg=-15.5,
            ),
        ),
    )
