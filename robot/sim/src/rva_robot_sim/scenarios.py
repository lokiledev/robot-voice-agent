"""What happens around the simulated Reachy Mini: a scene, and what its people do in it.

Scenarios are written as code, each a function, picked by its name in `SCENARIOS`, e.g.
`lune-sim --scenario greeting`. What the people do is a behavior tree of the steps of `steps.py`:
one after the other in a sequence, at once in a parallel. They wait for the robot to have answered
rather than for a time, so a run takes as long as the robot's answers do. One of them may be live,
someone at the real microphone speaking for them.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Final, Literal, Self

from py_trees.behaviour import Behaviour
from py_trees.common import ParallelPolicy
from py_trees.composites import Parallel, Sequence
from pydantic import model_validator

from rva_pydantic import RVABaseModel
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.scenes import Scene, at_the_front_desk, two_by_the_table
from rva_robot_sim.steps import (
    Play,
    RobotVoice,
    Say,
    Stage,
    Unstaged,
    Wait,
    WaitForRobot,
    WaitUntilRobotSpoke,
    alongside,
    anyway,
)

_SAID_S: Final = 4.0
"""How long a live person is given to say a line of theirs, where the script has one."""
_PAUSE_S: Final = 0.5
"""How long someone lets pass after the robot's last word before they speak."""
_ANSWERS_WITHIN_S: Final = 2.0
"""How long after a line the robot has to start answering it, before the scene goes on: it takes
about a second. Longer, and a line it does not answer leaves a gap."""
_GREETS_WITHIN_S: Final = 5.0
"""How long after someone starts turning to the robot it has to start greeting them: they
engaged 2 s into the turn, and were greeted a second later."""
_ANSWERS_A_NAME_WITHIN_S: Final = 4.0
"""The same after someone gave their name: the robot first checks for the name and remembers
the face, which took it 3 s."""


class Scenario(RVABaseModel):
    scene: Scene
    idle: dict[str, tuple[ClipName, ...]]
    """What each of the scene's people plays, one clip after the other, over and over, between
    their steps, by their name. None, and they stand in their A-pose."""
    live: str | None
    """Whom someone at the real microphone speaks for: what it hears opens their mouth and goes
    into the room the robot hears. They say no line of the script."""
    script: Callable[[Stage, RobotVoice], Behaviour]
    """What the people do, as a tree of steps on a stage, waiting on the robot's voice. Built
    anew each time: once to read it, once to play it."""

    def acts(self) -> tuple[Say | Play, ...]:
        """Every line and every clip of the script, in the order it is written."""
        tree = self.script(Unstaged(), RobotVoice())
        return tuple(step for step in tree.iterate() if isinstance(step, Say | Play))

    @model_validator(mode="after")
    def _everyone_in_the_scene(self) -> Self:
        names = sorted(person.name for person in self.scene.people)
        if sorted(self.idle) != names:
            raise ValueError(f"Idle clips for {sorted(self.idle)}, not the scene's {names}")
        if self.live is not None and self.live not in names:
            raise ValueError(f"{self.live} is live, and not in the scene's {names}")
        acts = self.acts()
        strangers = sorted({act.who for act in acts} - set(names))
        if strangers:
            raise ValueError(f"Steps for {strangers}, not in the scene's {names}")
        if any(isinstance(act, Say) and act.who == self.live for act in acts):
            raise ValueError(f"{self.live} is live, and the script has lines for them")
        return self


def still() -> Scenario:
    """Mia and Ben standing still and silent in their A-pose."""
    return Scenario(
        scene=two_by_the_table(),
        idle={"mia": (), "ben": ()},
        live=None,
        script=lambda stage, robot: Sequence("still", memory=True),
    )


def greeting(live: Literal["mia", "ben"] | None) -> Scenario:
    """Mia greets the robot, asks what it can do and for a joke; Ben listens.

    She waits for each of its answers before her next line. Both idle in between, and look
    around now and then.

    Args:
        live: Whom you speak for at the microphone. Mia: you replace her, and Ben listens now and
            then. Ben: you speak among her lines, and he only idles.
    """
    talk = ClipName.parse("static/f_gestic_talk_neutral_01")
    talk_more = ClipName.parse("static/f_gestic_talk_neutral_02")

    def script(stage: Stage, robot: RobotVoice) -> Behaviour:
        def asks(text: str, gesture: ClipName) -> Behaviour:
            said: Behaviour = (
                Wait(stage, _SAID_S) if live == "mia" else Say(stage, "mia", text, gesture)
            )
            answered = _heard_out(stage, robot, _ANSWERS_WITHIN_S)
            return Sequence(f"mia asks: {text}", memory=True, children=[said, answered])

        def listened_to(asking: Behaviour, clip: str) -> Behaviour:
            if live == "ben":
                return asking
            return alongside(asking, Play(stage, "ben", _clips(clip)))

        return Sequence(
            "greeting",
            memory=True,
            children=[
                Wait(stage, 3.0),
                listened_to(
                    asks("Hello Milo! How are you today?", talk),
                    "static/m_gestic_listen_neutral_01",
                ),
                asks("What can you do?", talk_more),
                listened_to(
                    asks("Can you tell me a joke?", talk), "static/m_gestic_listen_neutral_02"
                ),
                Wait(stage, _SAID_S)
                if live == "mia"
                else Say(stage, "mia", "Ha, that's a good one. Thank you!", talk_more),
            ],
        )

    return Scenario(
        scene=two_by_the_table(),
        idle={
            "mia": _clips("static/f_idle_neutral_01", "static/f_idle_look_around_01"),
            "ben": _clips("static/m_idle_neutral_01", "static/m_idle_look_around_01"),
        },
        live=live,
        script=script,
    )


def newcomer(live: Literal["mia"] | None) -> Scenario:
    """The robot's first day at the front desk: Mia and Ben try it out, then Cleo walks in.

    The story: the office has a new receptionist, a robot on the front desk. Mia wonders what to
    tell the visitor they expect, and Ben sends her to ask the robot, whose job it now is. They
    introduce themselves and have it rehearse its welcome speech, when the visitor walks in, Cleo,
    here for an interview; it greets her, and tells her who the two are.

    What it shows, in order: speech not said to the robot, while the head turns to whoever
    speaks; Mia engaging and giving her name, the first, who becomes the operator; Ben doing the
    same, then asking for a long speech, which Mia cuts short; Cleo walking in from out of view
    and waving, greeted and asked her name as an unknown face; then who is here, recognized.
    Start with an empty face database.

    Nobody says another's name to the robot, which takes it for the speaker's. What is said to
    it has no pause before its end, least of all after a name: "I'm Mia, the one who unboxed
    you" was heard as two utterances, the second cutting its answer to the first.

    Each turns to the robot and lets it greet them before speaking: a line said as they engage
    would cut the greeting off. Each then waits for its answer before the next line, but only
    a couple of seconds for it to start: a robot that says nothing leaves no long gap. Cleo
    starts walking once the speech is under way, and Mia cuts it short five seconds in, only if
    it still goes on then. Cleo listens to the robot with a clip that keeps her facing it, rather
    than idle and look away.

    Args:
        live: Whom you speak for at the microphone. Mia: you replace her lines, chatting with
            Ben, each given a few seconds, and still turn to the robot at her time.
    """

    def script(stage: Stage, robot: RobotVoice) -> Behaviour:
        def mia_says(text: str, gesture: str) -> Behaviour:
            if live == "mia":
                return Wait(stage, _SAID_S)
            return Say(stage, "mia", text, ClipName.parse(gesture))

        def greeted(within_s: float) -> Behaviour:
            return _heard_out(stage, robot, within_s)

        def answered(within_s: float) -> Behaviour:
            return _heard_out(stage, robot, within_s)

        def cleo_listens(until: Behaviour) -> Behaviour:
            # Her idle clip turns her head as far as 70 degrees, half the time over 15: she
            # would stop looking at the robot, and what she says next not be said to it. This
            # one keeps it within 10.
            return alongside(
                until, Play(stage, "cleo", _clips("static/f_gestic_listen_neutral_01"))
            )

        return Sequence(
            "newcomer",
            memory=True,
            children=[
                Wait(stage, 2.0),
                mia_says(
                    "We have a visitor at ten. What are we supposed to tell them?",
                    "static/f_gestic_talk_neutral_01",
                ),
                Wait(stage, 0.7),
                Say(
                    stage,
                    "ben",
                    "No idea. Ask the robot, welcoming people is its job now.",
                    ClipName.parse("static/m_gestic_talk_neutral_01"),
                ),
                Wait(stage, 0.7),
                mia_says(
                    "Good point. Let's see what it knows.",
                    "static/f_gestic_talk_neutral_02",
                ),
                Parallel(
                    "mia turns to the robot, which greets her",
                    ParallelPolicy.SuccessOnAll(),
                    [Play(stage, "mia", _clips("xyz/f_turn_right_60")), greeted(_GREETS_WITHIN_S)],
                ),
                mia_says(
                    "I'm Mia.",
                    "static/f_gestic_talk_relaxed_01",
                ),
                answered(_ANSWERS_A_NAME_WITHIN_S),
                Parallel(
                    "ben turns to the robot, which greets him",
                    ParallelPolicy.SuccessOnAll(),
                    [Play(stage, "ben", _clips("xyz/m_turn_left_60")), greeted(_GREETS_WITHIN_S)],
                ),
                # Two lines: said in one, the robot answers his name while he still speaks.
                Say(
                    stage,
                    "ben",
                    "I'm Ben and I fix whatever she breaks.",
                    ClipName.parse("static/m_gestic_talk_relaxed_01"),
                ),
                answered(_ANSWERS_A_NAME_WITHIN_S),
                Say(
                    stage,
                    "ben",
                    "Please give us the long version of your welcome speech for visitors.",
                    ClipName.parse("static/m_gestic_talk_neutral_01"),
                ),
                # The speech is made to Ben, and stops if he looks away: he keeps to the idle clip
                # that turns his head least, 24 degrees at most. His listening clips turn it 40.
                alongside(
                    Sequence(
                        "the speech, cut short as cleo walks in",
                        memory=True,
                        children=[
                            anyway(WaitUntilRobotSpoke(stage, robot, for_s=1.5, within_s=6.0)),
                            Parallel(
                                "cleo walks in, and mia cuts the speech short",
                                ParallelPolicy.SuccessOnAll(),
                                [
                                    Play(
                                        stage,
                                        "cleo",
                                        _clips(
                                            "xy/f_walk_start",
                                            "xy/f_walk_neutral_01",
                                            "xy/f_walk_stop",
                                            "xyz/f_turn_left_60",
                                            "static/f_wave_01",
                                        ),
                                    ),
                                    # Not said to a robot that has finished its speech already:
                                    # a short one is two sentences, ten seconds.
                                    anyway(
                                        Sequence(
                                            "mia cuts the speech short",
                                            memory=True,
                                            children=[
                                                WaitUntilRobotSpoke(
                                                    stage, robot, for_s=5.0, within_s=6.0
                                                ),
                                                mia_says(
                                                    "Hold that thought, here she comes.",
                                                    "static/f_gestic_talk_excited_01",
                                                ),
                                            ],
                                        )
                                    ),
                                ],
                            ),
                        ],
                    ),
                    Play(stage, "ben", _clips(*["static/m_idle_neutral_01"] * 3)),
                ),
                # She engages 4 to 8 s after her wave.
                cleo_listens(greeted(10.0)),
                Say(
                    stage,
                    "cleo",
                    "Hi, I'm Cleo, I'm here for an interview.",
                    ClipName.parse("static/f_gestic_talk_relaxed_02"),
                ),
                cleo_listens(answered(_ANSWERS_A_NAME_WITHIN_S)),
                Say(
                    stage,
                    "cleo",
                    "Thank you. And who are these two?",
                    ClipName.parse("static/f_gestic_talk_neutral_03"),
                ),
                cleo_listens(answered(_ANSWERS_WITHIN_S)),
                mia_says(
                    "Not bad for something I unboxed this morning. You're hired!",
                    "static/f_gestic_talk_excited_01",
                ),
            ],
        )

    return Scenario(
        scene=at_the_front_desk(),
        idle={
            "mia": _clips("static/f_idle_neutral_01", "static/f_idle_neutral_02"),
            "ben": _clips("static/m_idle_neutral_01", "static/m_idle_neutral_02"),
            "cleo": _clips("static/f_idle_neutral_03"),
        },
        live=live,
        script=script,
    )


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


def _heard_out(stage: Stage, robot: RobotVoice, within_s: float) -> Behaviour:
    """Wait for the robot to have said all it has to, then a pause; or go on if it says nothing.

    Args:
        stage: Whose clock the wait is on.
        robot: Whose reply is waited for.
        within_s: How long the robot has to start, before the scene goes on without it.
    """
    return anyway(
        Sequence(
            "the robot is heard out",
            memory=True,
            children=[WaitForRobot(stage, robot, within_s), Wait(stage, _PAUSE_S)],
        )
    )


def _clips(*names: str) -> tuple[ClipName, ...]:
    return tuple(ClipName.parse(name) for name in names)
