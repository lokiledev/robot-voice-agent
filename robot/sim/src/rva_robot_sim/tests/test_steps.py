import pytest
from py_trees.behaviour import Behaviour
from py_trees.common import ParallelPolicy, Status
from py_trees.composites import Parallel, Sequence

from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.scenarios import describe
from rva_robot_sim.steps import (
    Play,
    RobotVoice,
    Say,
    Wait,
    WaitForRobot,
    WaitUntilRobotSpoke,
    alongside,
    anyway,
)
from rva_schemas.record import RobotLine, RobotReplyEnded, RobotSentence

TALK = ClipName.parse("static/f_gestic_talk_neutral_01")
LISTEN = ClipName.parse("static/m_gestic_listen_neutral_01")
TICK_S = 0.1
LISTENING = (
    "f_gestic_listen_neutral_01",
    "m_idle_neutral_01+m_idle_neutral_01+m_idle_neutral_01",
    "cut",
)
SENTENCE = RobotSentence(text="Hello!")
LINE = RobotLine(text="Hello!", interrupted=False)
ENDED = RobotReplyEnded()


class FakeStage:
    """A line lasts half a second a word, a clip two seconds; what starts is logged."""

    def __init__(self) -> None:
        self.time = 0.0
        self.log: list[tuple[float, str, str]] = []
        self.busy_until: dict[str, float] = {}

    def now(self) -> float:
        return self.time

    def say(self, who: str, text: str, gesture: ClipName) -> float:
        return self._starts(who, text, 0.5 * len(text.split()))

    def play(self, who: str, clips: tuple[ClipName, ...]) -> float:
        return self._starts(who, "+".join(clip.name for clip in clips), 2.0 * len(clips))

    def cut(self, who: str) -> None:
        self._starts(who, "cut", 0.0)

    def _starts(self, who: str, what: str, lasts: float) -> float:
        self.log.append((self.time, who, what))
        # Listening is nothing the robot answers, nor waits the end of.
        if what not in LISTENING:
            self.busy_until[who] = self.time + lasts
        return lasts


def run(
    tree: Behaviour,
    stage: FakeStage,
    robot: RobotVoice,
    heard: dict[float, RobotSentence | RobotLine | RobotReplyEnded],
) -> Status:
    """Tick `tree` ten times a second until it ends, the robot heard at the times of `heard`."""
    for tick in range(3000):
        stage.time = tick * TICK_S
        if stage.time in heard:
            robot.hear(heard[stage.time], stage.time)
        tree.tick_once()
        if tree.status != Status.RUNNING:
            return tree.status
    raise AssertionError("still running after five minutes")


def test_a_step_starts_as_the_one_before_it_ends() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "chat",
        memory=True,
        children=[
            Say(stage, "mia", "One two three four", TALK),
            Wait(stage, 1.0),
            Say(stage, "ben", "Five six", TALK),
        ],
    )

    assert run(tree, stage, robot, {}) == Status.SUCCESS
    assert stage.log == [(0.0, "mia", "One two three four"), (3.0, "ben", "Five six")]


def test_the_next_line_waits_for_the_whole_reply_not_for_a_silence_in_it() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "greeting",
        memory=True,
        children=[
            Say(stage, "mia", "Hello Milo", TALK),
            WaitForRobot(stage, robot, within_s=8.0),
            Say(stage, "mia", "What can you do?", TALK),
        ],
    )

    # Two stretches of speech, silent for two seconds in between, while it calls a tool.
    heard = {2.5: SENTENCE, 4.0: LINE, 6.0: SENTENCE, 7.0: LINE, 7.5: ENDED}
    assert run(tree, stage, robot, heard) == Status.SUCCESS
    assert stage.log == [(0.0, "mia", "Hello Milo"), (7.5, "mia", "What can you do?")]


def test_a_robot_that_says_nothing_fails_the_scenario() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "greeting",
        memory=True,
        children=[
            Say(stage, "mia", "Hello Milo", TALK),
            WaitForRobot(stage, robot, within_s=8.0),
            Say(stage, "mia", "What can you do?", TALK),
        ],
    )

    assert run(tree, stage, robot, {}) == Status.FAILURE
    assert stage.log == [(0.0, "mia", "Hello Milo")]


def test_a_demo_goes_on_anyway_when_the_robot_says_nothing() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "greeting",
        memory=True,
        children=[
            Say(stage, "mia", "Hello Milo", TALK),
            anyway(WaitForRobot(stage, robot, within_s=8.0)),
            Say(stage, "mia", "What can you do?", TALK),
        ],
    )

    assert run(tree, stage, robot, {}) == Status.SUCCESS
    assert stage.log == [(0.0, "mia", "Hello Milo"), (9.0, "mia", "What can you do?")]


def test_a_robot_still_speaking_is_waited_for_past_the_time_it_had_to_start() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "story",
        memory=True,
        children=[
            WaitForRobot(stage, robot, within_s=8.0),
            Say(stage, "ben", "Thank you", TALK),
        ],
    )

    heard = {2.0: SENTENCE, 30.0: LINE, 30.5: ENDED}
    assert run(tree, stage, robot, heard) == Status.SUCCESS
    assert stage.log == [(30.5, "ben", "Thank you")]


def test_someone_talks_over_the_robot_once_it_has_spoken_long_enough() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "interruption",
        memory=True,
        children=[
            WaitUntilRobotSpoke(stage, robot, for_s=5.0, within_s=20.0),
            Say(stage, "mia", "Sorry, hold on", TALK),
        ],
    )

    # A short greeting first, which does not count: five seconds on end.
    heard: dict[float, RobotSentence | RobotLine | RobotReplyEnded] = {
        1.0: SENTENCE,
        2.0: LINE,
        4.0: SENTENCE,
    }
    assert run(tree, stage, robot, heard) == Status.SUCCESS
    assert stage.log == [(9.0, "mia", "Sorry, hold on")]


def test_what_plays_alongside_a_line_is_cut_when_the_line_ends() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "listening",
        memory=True,
        children=[
            alongside(
                Say(stage, "mia", "One two", TALK),
                Play(stage, "ben", (LISTEN, LISTEN, LISTEN)),
            ),
            Say(stage, "ben", "Three", TALK),
        ],
    )

    assert run(tree, stage, robot, {}) == Status.SUCCESS
    assert stage.log == [
        (0.0, "mia", "One two"),
        (
            0.0,
            "ben",
            "m_gestic_listen_neutral_01+m_gestic_listen_neutral_01+m_gestic_listen_neutral_01",
        ),
        (1.0, "ben", "cut"),
        (1.0, "ben", "Three"),
    ]


def test_steps_in_parallel_all_end_before_the_next() -> None:
    stage, robot = FakeStage(), RobotVoice()
    tree = Sequence(
        "turning",
        memory=True,
        children=[
            Parallel(
                "mia turns and is greeted",
                ParallelPolicy.SuccessOnAll(),
                [
                    Play(stage, "mia", (ClipName.parse("xyz/f_turn_right_60"),)),
                    WaitForRobot(stage, robot, within_s=8.0),
                ],
            ),
            Say(stage, "mia", "Hi Milo", TALK),
        ],
    )

    # The robot greets her while she turns.
    assert run(tree, stage, robot, {1.0: SENTENCE, 1.5: LINE, 2.5: ENDED}) == Status.SUCCESS
    assert stage.log == [(0.0, "mia", "f_turn_right_60"), (2.5, "mia", "Hi Milo")]


class ReactiveRobot:
    """Answers whoever acted last, a second after they end, once someone has turned to it.

    Each answer lasts `answer_s`; talked over, it stops at once.
    """

    def __init__(self, stage: FakeStage, voice: RobotVoice, answer_s: float | None) -> None:
        self._stage = stage
        self._voice = voice
        self._answer_s = answer_s
        self._until = 0.0
        self._answered = 0
        """How many of the stage's log entries were there when it last spoke."""
        self.talked_over: list[str] = []

    def tick(self) -> None:
        now = self._stage.time
        log = [entry for entry in self._stage.log if entry[2] not in LISTENING]
        if self._voice.speaking:
            said = [what for at, _, what in log[self._answered :] if " " in what]
            if said:
                self.talked_over += said
                self._answered = len(log)
                self._voice.hear(RobotLine(text="", interrupted=True), now)
                self._voice.hear(ENDED, now)
            elif now >= self._until:
                self._voice.hear(LINE, now)
                self._voice.hear(ENDED, now)
            return
        if self._answer_s is None or not any("turn" in what for _, _, what in log):
            return
        if len(log) > self._answered and now >= max(self._stage.busy_until.values()) + 1.0:
            self._answered = len(log)
            self._until = now + self._answer_s
            self._voice.hear(SENTENCE, now)


HOLD_ON = "Sorry, hold on. Someone's coming."


@pytest.mark.parametrize(
    ("answer_s", "cut_short", "ends_at"),
    [(4.0, False, 92.5), (20.0, True, 192.9), (None, False, 87.5)],
    ids=["short answers", "long answers", "a silent robot"],
)
def test_newcomer_keeps_its_order_however_long_the_robot_answers(
    answer_s: float | None, cut_short: bool, ends_at: float
) -> None:
    stage, voice = FakeStage(), RobotVoice()
    robot = ReactiveRobot(stage, voice, answer_s)
    tree = describe("newcomer").script(stage, voice)

    for tick in range(3000):
        stage.time = tick * TICK_S
        robot.tick()
        tree.tick_once()
        if tree.status != Status.RUNNING:
            break

    assert tree.status == Status.SUCCESS
    acts = [
        ("mia", "So, did you get the demo working in the end?"),
        ("ben", "Almost. The head still wobbles when it turns."),
        ("mia", "Look, it's watching us. Let's say hi."),
        ("mia", "f_turn_right_60"),
        ("mia", "Hi Milo! I'm Mia."),
        ("ben", "m_turn_left_60"),
        ("ben", "And I'm Ben."),
        ("ben", "Can you tell us a story about a robot who learns to dance?"),
        ("cleo", "f_walk_start+f_walk_neutral_01+f_walk_stop+f_turn_left_60+f_wave_01"),
        ("mia", HOLD_ON),
        ("cleo", "Hi, I'm Cleo. I just joined the team."),
        ("cleo", "Nice to meet you. Who else is here?"),
        ("ben", "Right, back to work. See you later!"),
    ]
    # Mia cuts the story short only if it still goes on: nobody else talks over the robot.
    assert [(who, what) for _, who, what in stage.log if what not in LISTENING] == [
        act for act in acts if cut_short or act != ("mia", HOLD_ON)
    ]
    assert robot.talked_over == ([HOLD_ON] if cut_short else [])
    assert stage.time == pytest.approx(ends_at)
