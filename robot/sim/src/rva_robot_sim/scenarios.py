"""What happens around the simulated Reachy Mini: a scene, and a script for each of its people.

Scenarios are written as code, each a function, picked by its name in `SCENARIOS`, e.g.
`lune-sim --scenario greeting`. A person's script is scripted, saying lines with gestures at
given times, or live, someone at the real microphone speaking for them. Steps happen at fixed
times of the simulation: nothing waits for the robot.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Annotated, Final, Literal

from pydantic import Field, field_validator, model_validator

from rva_pydantic import RVABaseModel
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.scenes import Scene, chatting_by_the_table, two_by_the_table


class Say(RVABaseModel):
    """Say a line, gesturing as long as it lasts: repeated if shorter, cut where it ends."""

    kind: Literal["say"] = "say"
    at_s: float = Field(ge=0)
    """When they start, in seconds since the simulation started."""
    text: str
    """One or a few sentences, in their voice."""
    gesture: ClipName


class Play(RVABaseModel):
    """Play clips once, one after the other, each from where the one before left them.

    A walk is several: starting, walking, stopping, turning to face someone; idling in between
    would stop them on the way.
    """

    kind: Literal["play"] = "play"
    at_s: float = Field(ge=0)
    """When it starts, in seconds since the simulation started."""
    clips: tuple[ClipName, ...] = Field(min_length=1)


type ScriptedStep = Annotated[Say | Play, Field(discriminator="kind")]


class Scripted(RVABaseModel):
    """What they do and say, step by step; their mouth follows their lines."""

    kind: Literal["scripted"] = "scripted"
    idle: tuple[ClipName, ...]
    """Played one after the other, over and over, between the steps. None, and they stand in
    their A-pose."""
    steps: tuple[ScriptedStep, ...]
    """In order; each must end before the next starts."""

    @field_validator("steps")
    @classmethod
    def _in_order(cls, steps: tuple[ScriptedStep, ...]) -> tuple[ScriptedStep, ...]:
        return _in_order(steps)


class Live(RVABaseModel):
    """Someone at the real microphone speaks for them; the steps only move them.

    What the microphone hears opens their mouth and goes into the room the robot hears.
    """

    kind: Literal["live"] = "live"
    idle: tuple[ClipName, ...]
    """Played one after the other, over and over, between the steps."""
    steps: tuple[Play, ...]
    """In order; each must end before the next starts."""

    @field_validator("steps")
    @classmethod
    def _in_order(cls, steps: tuple[Play, ...]) -> tuple[Play, ...]:
        return _in_order(steps)


type Script = Annotated[Scripted | Live, Field(discriminator="kind")]


class Scenario(RVABaseModel):
    scene: Scene
    scripts: dict[str, Script]
    """Everyone's in the scene, by their name. One at most is live: there is one microphone."""

    @model_validator(mode="after")
    def _everyone_once(self) -> "Scenario":
        names = [person.name for person in self.scene.people]
        if sorted(self.scripts) != sorted(names):
            raise ValueError(f"Scripts for {sorted(self.scripts)}, not the scene's {names}")
        live = sorted(name for name, script in self.scripts.items() if isinstance(script, Live))
        if len(live) > 1:
            raise ValueError(f"{live} are live; one microphone speaks for one at most")
        return self


def still() -> Scenario:
    """Mia and Ben standing still and silent in their A-pose."""
    return Scenario(
        scene=two_by_the_table(),
        scripts={
            "mia": Scripted(idle=(), steps=()),
            "ben": Scripted(idle=(), steps=()),
        },
    )


def greeting(live: Literal["mia", "ben"] | None) -> Scenario:
    """Mia greets the robot, asks what it can do and for a joke; Ben listens.

    Her lines are twenty seconds apart, leaving the robot time to answer. Both idle in between,
    and look around now and then.

    Args:
        live: Whom you speak for at the microphone. Mia: you replace her, and only gesture with
            your own words. Ben: you speak among her lines, and he only idles.
    """
    mia_idle = _clips("static/f_idle_neutral_01", "static/f_idle_look_around_01")
    ben_idle = _clips("static/m_idle_neutral_01", "static/m_idle_look_around_01")
    talk = ClipName.parse("static/f_gestic_talk_neutral_01")
    talk_more = ClipName.parse("static/f_gestic_talk_neutral_02")
    mia: Script = Scripted(
        idle=mia_idle,
        steps=(
            Say(at_s=3.0, text="Hello Milo! How are you today?", gesture=talk),
            Say(at_s=23.0, text="What can you do?", gesture=talk_more),
            Say(at_s=43.0, text="Can you tell me a joke?", gesture=talk),
            Say(at_s=63.0, text="Ha, that's a good one. Thank you!", gesture=talk_more),
        ),
    )
    ben: Script = Scripted(
        idle=ben_idle,
        steps=(
            Play(at_s=4.0, clips=_clips("static/m_gestic_listen_neutral_01")),
            Play(at_s=44.0, clips=_clips("static/m_gestic_listen_neutral_02")),
        ),
    )
    match live:
        case "mia":
            mia = Live(idle=mia_idle, steps=())
        case "ben":
            ben = Live(idle=ben_idle, steps=())
        case None:
            pass
    return Scenario(scene=two_by_the_table(), scripts={"mia": mia, "ben": ben})


def newcomer(live: Literal["mia"] | None) -> Scenario:
    """Mia and Ben chat, then introduce themselves to the robot; Cleo walks in, a stranger.

    What it shows, in order: speech not said to the robot, while the head turns to whoever
    speaks; Mia engaging and giving her name, the first, who becomes the operator; Ben asking for
    a story and Mia cutting it short; Cleo walking in from out of view and waving, greeted and
    asked her name as an unknown face; then who is here, recognized. Lines are at fixed times,
    tuned for the robot's answers to fit between them, with the LLM fed the events expected;
    start with an empty face database. Each turns to the robot and lets it greet them before
    speaking: a line said as they engage would cut the greeting off. Cleo gives her name soon
    after she engages, greeted or not: kept waiting, she looks away and disengages.

    Args:
        live: Whom you speak for at the microphone. Mia: you replace her lines, chatting with
            Ben, and still turn to the robot at her time.
    """
    mia_idle = _clips("static/f_idle_neutral_01", "static/f_idle_neutral_02")
    mia: Script = Scripted(
        idle=mia_idle,
        steps=(
            Say(
                at_s=2.0,
                text="So, did you get the demo working in the end?",
                gesture=ClipName.parse("static/f_gestic_talk_neutral_01"),
            ),
            Say(
                at_s=10.0,
                text="Look, it's watching us. Let's say hi.",
                gesture=ClipName.parse("static/f_gestic_talk_neutral_02"),
            ),
            Play(at_s=13.0, clips=_clips("xyz/f_turn_right_60")),
            Say(
                at_s=19.5,
                text="Hi Milo! I'm Mia.",
                gesture=ClipName.parse("static/f_gestic_talk_relaxed_01"),
            ),
            Say(
                at_s=50.0,
                text="Sorry, hold on. Someone's coming.",
                gesture=ClipName.parse("static/f_gestic_talk_excited_01"),
            ),
        ),
    )
    ben: Script = Scripted(
        idle=_clips("static/m_idle_neutral_01", "static/m_idle_neutral_02"),
        steps=(
            Say(
                at_s=6.0,
                text="Almost. The head still wobbles when it turns.",
                gesture=ClipName.parse("static/m_gestic_talk_neutral_01"),
            ),
            Play(at_s=26.0, clips=_clips("xyz/m_turn_left_60")),
            Say(
                at_s=34.0,
                text="And I'm Ben. Can you tell us a story about a robot who learns to dance?",
                gesture=ClipName.parse("static/m_gestic_talk_relaxed_01"),
            ),
            Say(
                at_s=77.0,
                text="Right, back to work. See you later!",
                gesture=ClipName.parse("static/m_gestic_talk_neutral_02"),
            ),
        ),
    )
    cleo: Script = Scripted(
        idle=_clips("static/f_idle_neutral_03"),
        steps=(
            Play(
                at_s=41.0,
                clips=_clips(
                    "xy/f_walk_start",
                    "xy/f_walk_neutral_01",
                    "xy/f_walk_stop",
                    "xyz/f_turn_left_60",
                    "static/f_wave_01",
                ),
            ),
            Say(
                at_s=58.0,
                text="Hi, I'm Cleo. I just joined the team.",
                gesture=ClipName.parse("static/f_gestic_talk_relaxed_02"),
            ),
            Say(
                at_s=67.0,
                text="Nice to meet you. Who else is here?",
                gesture=ClipName.parse("static/f_gestic_talk_neutral_03"),
            ),
        ),
    )
    match live:
        case "mia":
            mia = Live(idle=mia_idle, steps=(Play(at_s=13.0, clips=_clips("xyz/f_turn_right_60")),))
        case None:
            pass
    return Scenario(scene=chatting_by_the_table(), scripts={"mia": mia, "ben": ben, "cleo": cleo})


SCENARIOS: Final[Mapping[str, Callable[[], Scenario]]] = {
    "still": still,
    "greeting": functools.partial(greeting, live=None),
    "greeting_as_mia": functools.partial(greeting, live="mia"),
    "greeting_as_ben": functools.partial(greeting, live="ben"),
    "newcomer": functools.partial(newcomer, live=None),
    "newcomer_as_mia": functools.partial(newcomer, live="mia"),
}
"""Every scenario, by the name `--scenario` takes."""


def describe(name: str) -> Scenario:
    """The scenario `name` describes.

    Raises:
        KeyError: If there is no such scenario; its message lists them.
    """
    if name not in SCENARIOS:
        raise KeyError(f"No scenario {name!r}; there are {sorted(SCENARIOS)}")
    return SCENARIOS[name]()


def _in_order[Step: Say | Play](steps: tuple[Step, ...]) -> tuple[Step, ...]:
    starts = [step.at_s for step in steps]
    if starts != sorted(starts):
        raise ValueError(f"Steps start at {starts}, not in order")
    return steps


def _clips(*names: str) -> tuple[ClipName, ...]:
    return tuple(ClipName.parse(name) for name in names)
