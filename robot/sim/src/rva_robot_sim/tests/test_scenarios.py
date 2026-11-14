from pathlib import Path

import pytest
from pydantic import ValidationError

from rva_pydantic import load_yaml
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.config import SimConfig
from rva_robot_sim.scenarios import SCENARIOS, Live, Play, Say, Scenario, Scripted, describe
from rva_robot_sim.scenes import Person, Scene, two_by_the_table

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"
TALK = ClipName.parse("static/f_gestic_talk_neutral_01")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_scripts_everyone_in_its_scene(name: str) -> None:
    scenario = describe(name)

    assert sorted(scenario.scripts) == sorted(person.name for person in scenario.scene.people)
    assert all(person.avatar.startswith("Adults/") for person in scenario.scene.people)


def test_ana_greets_the_robot_while_ben_listens() -> None:
    scenario = describe("greeting")

    ana, ben = scenario.scripts["ana"], scenario.scripts["ben"]
    assert isinstance(ana, Scripted) and isinstance(ben, Scripted)
    assert [(step.at_s, step.text) for step in ana.steps if isinstance(step, Say)] == [
        (3.0, "Hello Reachy! How are you today?"),
        (23.0, "What can you do?"),
        (43.0, "Can you tell me a joke?"),
        (63.0, "Ha, that's a good one. Thank you!"),
    ]
    assert not any(isinstance(step, Say) for step in ben.steps)
    assert scenario.scene.person("ana").voice != scenario.scene.person("ben").voice


@pytest.mark.parametrize(
    ("name", "live", "scripted"),
    [("greeting_as_ana", "ana", "ben"), ("greeting_as_ben", "ben", "ana")],
)
def test_you_speak_for_one_of_them_while_the_other_keeps_their_script(
    name: str, live: str, scripted: str
) -> None:
    scenario = describe(name)

    assert scenario.scripts[live] == Live(idle=scenario.scripts[live].idle, steps=())
    assert scenario.scripts[scripted] == describe("greeting").scripts[scripted]


def test_still_people_stand_silent_in_their_a_pose() -> None:
    assert set(describe("still").scripts.values()) == {Scripted(idle=(), steps=())}


def test_a_live_script_cannot_say_lines() -> None:
    with pytest.raises(ValidationError, match="play"):
        Live.model_validate({"idle": [], "steps": [{"kind": "say", "at_s": 1.0, "text": "Hi"}]})


def test_steps_are_written_in_order() -> None:
    with pytest.raises(ValidationError, match="not in order"):
        Scripted(
            idle=(), steps=(Play(at_s=10.0, clip=TALK), Say(at_s=5.0, text="Hi", gesture=TALK))
        )


def test_everyone_in_the_scene_has_a_script_and_nobody_else() -> None:
    script = Scripted(idle=(), steps=())

    with pytest.raises(ValidationError, match=r"Scripts for \['ana', 'bob'\], not the scene's"):
        Scenario(scene=two_by_the_table(), scripts={"ana": script, "bob": script})


def test_one_microphone_speaks_for_one_person_at_most() -> None:
    live = Live(idle=(), steps=())

    with pytest.raises(ValidationError, match="one microphone speaks for one at most"):
        Scenario(scene=two_by_the_table(), scripts={"ana": live, "ben": live})


def test_names_in_a_scene_are_unique() -> None:
    ana = two_by_the_table().person("ana")

    with pytest.raises(ValidationError, match="each name once"):
        Scene(table_height=1.0, people=(ana, ana))


def test_a_person_is_found_by_name() -> None:
    scene = two_by_the_table()

    assert scene.person("ben") == Person(
        name="ben",
        avatar="Adults/Male_Adult_04",
        voice="am_michael",
        position=(0.8, -0.65),
        turned_deg=0.0,
    )
    with pytest.raises(KeyError, match=r"Nobody named 'cleo'; there are \['ana', 'ben'\]"):
        scene.person("cleo")


def test_an_unknown_scenario_lists_the_others() -> None:
    with pytest.raises(KeyError, match="greeting_as_ben"):
        describe("party")


def test_the_reachy_profile_plays_the_greeting() -> None:
    assert load_yaml(SimConfig, SIM_YAML).scenario == "greeting"


def test_a_profile_cannot_name_a_scenario_that_does_not_exist(tmp_path: Path) -> None:
    text = SIM_YAML.read_text()
    assert "scenario: greeting\n" in text
    misnamed = tmp_path / "sim.yaml"
    misnamed.write_text(text.replace("scenario: greeting\n", "scenario: party\n"))

    with pytest.raises(ValidationError, match="No scenario 'party'"):
        load_yaml(SimConfig, misnamed)
