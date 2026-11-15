"""What the simulated Reachy Mini has around it at the start: its table and the people.

Scenes are written as code, each a function; a scenario (`scenarios.py`) picks one and says what
happens in it.
"""

from typing import Final

from pydantic import field_validator

from rva_pydantic import RVABaseModel

_LAMPS: Final = tuple((x, y, 2.6) for x in (-2.0, 1.0, 4.0, 7.0, 10.0) for y in (-3.0, -0.5))
"""Over the reception's counter and down its lounge, under the ceiling."""


class Prop(RVABaseModel):
    """Something a person carries, seen and not touched: one glTF model."""

    name: str
    """What the model is kept as once downloaded, e.g. `pizzas`."""
    url: str
    """Where the model is downloaded from on first use, as a binary glTF."""
    sha256: str
    """Of the download, in hexadecimal."""
    scale: float
    """Meters per unit of the model."""


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
    carries: Prop | None
    """What they hold level on their left forearm, which then keeps still under it; None, and
    their arms are the clips'."""


class Interior(RVABaseModel):
    """A furnished room around the robot, seen and not touched: one glTF model.

    The robot stands on a piece of its furniture, which the scene's `table_height` is the height
    of.
    """

    name: str
    """What the model is kept as once downloaded, e.g. `reception`."""
    url: str
    """Where the model is downloaded from on first use, as a binary glTF."""
    sha256: str
    """Of the download, in hexadecimal."""
    scale: float
    """Meters per unit of the model: 0.01 for one in centimeters."""
    robot_at: tuple[float, float]
    """Where the robot stands in the model, in meters: along the model's x, and along its -z,
    which Genesis makes y as it stands a glTF's y axis up."""
    robot_faces_deg: float
    """Which way the robot faces there, in degrees, counterclockwise seen from above; 0 faces
    the model's +x."""
    lamps: tuple[tuple[float, float, float], ...]
    """Where the lamps that light the room hang, in meters, in the robot's frame: a closed room
    gets no light from the panorama around the world."""
    lamp_lumens: float
    """How much light each lamp gives."""


class Scene(RVABaseModel):
    table_height: float
    """Of the table top the robot stands on, in meters. The head camera sits about 0.15 m above
    it."""
    people: tuple[Person, ...]
    interior: Interior | None
    """The room all of it is in. None, and the robot's table stands on a bare floor under the
    panorama's sky."""

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
                carries=None,
            ),
            Person(
                name="ben",
                avatar="Adults/Male_Adult_04",
                voice="am_michael",
                position=(0.8, -0.65),
                turned_deg=0.0,
                carries=None,
            ),
        ),
        interior=None,
    )


def at_the_front_desk() -> Scene:
    """The robot on a reception's counter, Mia and Ben chatting in front, Cleo out of its view.

    The room is "Luiz IEDA - Recepção" by SrMonteiro, CC BY 4.0, from Sketchfab through
    Objaverse: a reception with a lounge. The robot stands at the end of its counter, 1.08 m
    high, and looks down the lounge.

    Mia and Ben stand 1.2 m away and 0.5 m apart, each turned 60 degrees to the other: their
    faces are seen, but not looking at the robot. A 60-degree turn clip faces each of them to it.
    Where they stand is what lets them engage in time: at 1.5 m, Mia was not engaged when she
    gave her name, and 18 degrees off the robot's axis, her gaze read as looking away. Cleo stands
    3.6 m to the robot's right, 71 degrees off its axis and out of view, facing its left:
    Rocketbox's walk clips take her to (1.3, -0.75), 30 degrees off its axis, where a 60-degree
    turn to her left faces her to it; at 34 degrees, her name was not heard as said to the robot.
    She starts between a plant and an armchair, and carries four pizzas: "Pizza Box" by
    ManuMagana, CC BY 4.0, from Sketchfab through Objaverse.
    """
    return Scene(
        table_height=1.08,
        people=(
            Person(
                name="mia",
                avatar="Adults/Female_Adult_01",
                voice="af_bella",
                position=(1.2, 0.3),
                turned_deg=60.0,
                carries=None,
            ),
            Person(
                name="ben",
                avatar="Adults/Male_Adult_04",
                voice="am_michael",
                position=(1.2, -0.2),
                turned_deg=-60.0,
                carries=None,
            ),
            Person(
                name="cleo",
                avatar="Adults/Female_Adult_05",
                voice="bf_emma",
                position=(1.24, -3.6),
                # Facing the robot's left, along +y.
                turned_deg=-19.0,
                carries=Prop(
                    name="pizzas",
                    url="https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-101/f671b486002848adbfe2f3771f3266e3.glb",
                    sha256="418389212904f3cd325f780cb9fbd9e28981f1ee32c60bd39e03cf91137c0c13",
                    # Four boxes 0.3 m wide, stacked as high.
                    scale=0.45,
                ),
            ),
        ),
        interior=Interior(
            name="reception",
            url="https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-152/d28ffb65c0be48e2b1821e649be0984e.glb",
            sha256="5c12dea2c86d0a19464f7467090b49c01649843b64e4220f6a8125167464b849",
            scale=0.01,
            robot_at=(171.5, -9.28),
            robot_faces_deg=-90.0,
            lamps=_LAMPS,
            lamp_lumens=18.0,
        ),
    )
