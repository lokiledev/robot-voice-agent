from collections.abc import Iterator
from uuid import uuid4

import kickmsg
import pytest

from rva_robot_sim.bus import RobotBus
from rva_robot_sim.npc import (
    PersonPose,
    ScenarioStart,
    SimulationNotRunningError,
    lateral,
    open_people_mover,
    start_scenario,
)
from rva_robot_sim.robot_speech import RobotSpeechPublisher, decode
from rva_schemas.record import AgentStarted, RobotLine, RobotReplyEnded, RobotSentence


@pytest.fixture
def namespace() -> Iterator[str]:
    name = f"rva_test_{uuid4().hex[:12]}"
    yield name
    kickmsg.Registry.unlink(name)


@pytest.mark.parametrize(
    ("elapsed", "y"),
    [
        (0.0, 0.0),
        # 0.8 m each side at 0.4 m/s: 2 s to an end, 8 s there and back.
        (1.0, -0.4),
        (2.0, -0.8),
        (3.0, -0.4),
        (4.0, 0.0),
        (6.0, 0.8),
        (7.0, 0.4),
        (8.0, 0.0),
        (10.0, -0.8),
    ],
)
def test_pacing_sideways_goes_to_the_robots_right_first(elapsed: float, y: float) -> None:
    assert lateral((1.1, 0.0), half_width=0.8, speed=0.4, elapsed=elapsed) == pytest.approx(
        (1.1, y)
    )


def test_poses_reach_the_simulation_in_order(namespace: str) -> None:
    assert open_people_mover(namespace, "test_walk") is None
    first = PersonPose(person="mia", position=(1.1, 0.0), turned_deg=0.0)
    second = PersonPose(person="mia", position=(1.1, -0.1), turned_deg=10.0)

    bus = RobotBus(namespace, (4, 2))
    try:
        mover = open_people_mover(namespace, "test_walk")
        assert mover is not None
        mover.send(first)
        mover.send(second)
        poses = bus.poses()
    finally:
        bus.close()

    assert poses == [first, second]
    assert open_people_mover(namespace, "test_walk") is None


def test_the_scenario_is_started_through_the_simulations_mailbox(namespace: str) -> None:
    with pytest.raises(SimulationNotRunningError, match="No simulation in kickmsg namespace"):
        start_scenario(namespace, "lune")

    bus = RobotBus(namespace, (4, 2))
    try:
        start_scenario(namespace, "lune")
        start_scenario(namespace, "npc start")
        starts = bus.starts()
    finally:
        bus.close()

    assert starts == [ScenarioStart(by="lune"), ScenarioStart(by="npc start")]


def test_what_the_robot_says_reaches_the_simulation_and_nothing_else_does(namespace: str) -> None:
    sentence = RobotSentence(text="Hello Mia!")
    line = RobotLine(text="Hello Mia!", interrupted=False)

    bus = RobotBus(namespace, (4, 2))
    try:
        agent = RobotSpeechPublisher(namespace)
        agent.record(AgentStarted(name="Milo", tools=()))
        agent.record(sentence)
        agent.record(line)
        agent.record(RobotReplyEnded())
        heard = bus.robot_speech()
    finally:
        bus.close()

    assert heard == [sentence, line, RobotReplyEnded()]


def test_the_robots_speech_is_read_from_its_json() -> None:
    assert decode(b'{"kind":"robot_sentence","text":"Hi."}') == RobotSentence(text="Hi.")
    assert decode(b'{"kind":"robot_line","text":"Hi.","interrupted":true}') == RobotLine(
        text="Hi.", interrupted=True
    )
    assert decode(b'{"kind":"robot_reply_ended"}') == RobotReplyEnded()
    with pytest.raises(ValueError, match="user_line"):
        decode(b'{"kind":"user_line","text":"Hi.","speaker":null}')
