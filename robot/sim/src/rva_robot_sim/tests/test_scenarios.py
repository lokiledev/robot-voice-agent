import math
from pathlib import Path

import pytest
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence
from pydantic import ValidationError

from rva_pydantic import load_yaml
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.config import SimConfig
from rva_robot_sim.scenarios import SCENARIOS, Scenario, describe
from rva_robot_sim.scenes import Person, Scene, at_the_front_desk, two_by_the_table
from rva_robot_sim.steps import Play, RobotVoice, Say, Stage

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"
TALK = ClipName.parse("static/f_gestic_talk_neutral_01")


def _lines(scenario: Scenario) -> list[tuple[str, str]]:
    return [(act.who, act.text) for act in scenario.acts() if isinstance(act, Say)]


def _plays(scenario: Scenario) -> list[tuple[str, list[str]]]:
    return [
        (act.who, [clip.name for clip in act.clips])
        for act in scenario.acts()
        if isinstance(act, Play)
    ]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_has_idle_clips_for_everyone_in_its_scene(name: str) -> None:
    scenario = describe(name)

    assert sorted(scenario.idle) == sorted(person.name for person in scenario.scene.people)
    assert all(person.avatar.startswith("Adults/") for person in scenario.scene.people)


def test_mia_greets_the_robot_while_ben_listens() -> None:
    scenario = describe("greeting")

    assert _lines(scenario) == [
        ("mia", "Hello Milo! How are you today?"),
        ("mia", "What can you do?"),
        ("mia", "Can you tell me a joke?"),
        ("mia", "Ha, that's a good one. Thank you!"),
    ]
    assert _plays(scenario) == [
        ("ben", ["m_gestic_listen_neutral_01"]),
        ("ben", ["m_gestic_listen_neutral_02"]),
    ]
    assert scenario.live is None
    assert scenario.scene.person("mia").voice != scenario.scene.person("ben").voice


def test_you_speak_for_mia_while_ben_still_listens() -> None:
    scenario = describe("greeting_as_mia")

    assert scenario.live == "mia"
    assert _lines(scenario) == []
    assert _plays(scenario) == _plays(describe("greeting"))


def test_you_speak_for_ben_among_mias_lines() -> None:
    scenario = describe("greeting_as_ben")

    assert scenario.live == "ben"
    assert _lines(scenario) == _lines(describe("greeting"))
    assert _plays(scenario) == []


def test_mia_and_ben_chat_then_meet_the_robot_and_cleo_walks_in() -> None:
    scenario = describe("newcomer")

    assert _lines(scenario) == [
        ("mia", "Did you order the pizzas?"),
        ("ben", "Twenty minutes ago. I could eat the box."),
        ("mia", "Then come and meet the new receptionist while we wait."),
        ("mia", "I'm Mia."),
        ("ben", "I'm Ben and I'm very hungry."),
        ("ben", "Tell us a long story to pass the time."),
        ("mia", "Hold on, I think that's dinner."),
        ("cleo", "Hi, I'm Cleo, I have four pizzas."),
        ("cleo", "Thank you. Is Ben around?"),
        ("ben", "Best receptionist we ever had."),
    ]
    assert _plays(scenario) == [
        ("mia", ["f_turn_right_60"]),
        ("ben", ["m_turn_left_60"]),
        (
            "cleo",
            ["f_walk_start", "f_walk_neutral_01", "f_walk_stop", "f_turn_left_60", "f_wave_01"],
        ),
        # Ben holds still while the story is told to him.
        ("ben", ["m_idle_neutral_01", "m_idle_neutral_01", "m_idle_neutral_01"]),
        # While the robot greets her, then answers each of her lines.
        ("cleo", ["f_gestic_listen_neutral_01"]),
        ("cleo", ["f_gestic_listen_neutral_01"]),
        ("cleo", ["f_gestic_listen_neutral_01"]),
    ]


def test_mia_and_ben_face_each_other_and_cleo_starts_out_of_the_robots_view() -> None:
    scene = at_the_front_desk()

    assert [(p.name, p.turned_deg) for p in scene.people[:2]] == [("mia", 60.0), ("ben", -60.0)]
    x, y = scene.person("cleo").position
    # The head camera sees 40 degrees each side of its axis.
    assert math.degrees(math.atan2(-y, x)) > 70.0
    assert len({person.voice for person in scene.people}) == 3


def test_you_speak_for_mia_who_still_turns_to_the_robot() -> None:
    scenario = describe("newcomer_as_mia")

    assert scenario.live == "mia"
    assert _lines(scenario) == [line for line in _lines(describe("newcomer")) if line[0] != "mia"]
    assert _plays(scenario) == _plays(describe("newcomer"))


def test_still_people_stand_silent_in_their_a_pose() -> None:
    scenario = describe("still")

    assert scenario.idle == {"mia": (), "ben": ()}
    assert scenario.acts() == ()


def _nothing(stage: Stage, robot: RobotVoice) -> Behaviour:
    return Sequence("nothing", memory=True)


def _mia_says_hi(stage: Stage, robot: RobotVoice) -> Behaviour:
    return Sequence("hi", memory=True, children=[Say(stage, "mia", "Hi", TALK)])


def test_everyone_in_the_scene_has_idle_clips_and_nobody_else() -> None:
    with pytest.raises(ValidationError, match=r"Idle clips for \['bob', 'mia'\], not the scene's"):
        Scenario(scene=two_by_the_table(), idle={"mia": (), "bob": ()}, live=None, script=_nothing)


def test_the_live_person_is_one_of_the_scene() -> None:
    with pytest.raises(ValidationError, match="cleo is live, and not in the scene's"):
        Scenario(
            scene=two_by_the_table(), idle={"mia": (), "ben": ()}, live="cleo", script=_nothing
        )


def test_the_script_has_no_line_for_the_live_person() -> None:
    with pytest.raises(ValidationError, match="mia is live, and the script has lines for them"):
        Scenario(
            scene=two_by_the_table(), idle={"mia": (), "ben": ()}, live="mia", script=_mia_says_hi
        )


def test_the_script_only_has_steps_for_people_of_the_scene() -> None:
    def script(stage: Stage, robot: RobotVoice) -> Behaviour:
        return Sequence("wave", memory=True, children=[Play(stage, "cleo", (TALK,))])

    with pytest.raises(ValidationError, match=r"Steps for \['cleo'\], not in the scene's"):
        Scenario(scene=two_by_the_table(), idle={"mia": (), "ben": ()}, live=None, script=script)


def test_names_in_a_scene_are_unique() -> None:
    mia = two_by_the_table().person("mia")

    with pytest.raises(ValidationError, match="each name once"):
        Scene(table_height=1.0, people=(mia, mia), interior=None)


def test_a_person_is_found_by_name() -> None:
    scene = two_by_the_table()

    assert scene.person("ben") == Person(
        name="ben",
        avatar="Adults/Male_Adult_04",
        voice="am_michael",
        position=(0.8, -0.65),
        turned_deg=0.0,
        carries=None,
    )
    with pytest.raises(KeyError, match=r"Nobody named 'cleo'; there are \['mia', 'ben'\]"):
        scene.person("cleo")


def test_an_unknown_scenario_lists_the_others() -> None:
    with pytest.raises(KeyError, match="greeting_as_ben"):
        describe("party")


def test_the_reachy_profile_plays_the_newcomer() -> None:
    assert load_yaml(SimConfig, SIM_YAML).scenario == "newcomer"


def test_a_profile_cannot_name_a_scenario_that_does_not_exist(tmp_path: Path) -> None:
    text = SIM_YAML.read_text()
    assert "scenario: newcomer\n" in text
    misnamed = tmp_path / "sim.yaml"
    misnamed.write_text(text.replace("scenario: newcomer\n", "scenario: party\n"))

    with pytest.raises(ValidationError, match="No scenario 'party'"):
        load_yaml(SimConfig, misnamed)
