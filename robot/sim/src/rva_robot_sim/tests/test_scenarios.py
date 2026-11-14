import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from rva_pydantic import load_yaml
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.config import SimConfig
from rva_robot_sim.scenarios import SCENARIOS, Live, Play, Say, Scenario, Scripted, describe
from rva_robot_sim.scenes import Person, Scene, chatting_by_the_table, two_by_the_table

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"
TALK = ClipName.parse("static/f_gestic_talk_neutral_01")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_scripts_everyone_in_its_scene(name: str) -> None:
    scenario = describe(name)

    assert sorted(scenario.scripts) == sorted(person.name for person in scenario.scene.people)
    assert all(person.avatar.startswith("Adults/") for person in scenario.scene.people)


def test_mia_greets_the_robot_while_ben_listens() -> None:
    scenario = describe("greeting")

    mia, ben = scenario.scripts["mia"], scenario.scripts["ben"]
    assert isinstance(mia, Scripted) and isinstance(ben, Scripted)
    assert [(step.at_s, step.text) for step in mia.steps if isinstance(step, Say)] == [
        (3.0, "Hello Milo! How are you today?"),
        (23.0, "What can you do?"),
        (43.0, "Can you tell me a joke?"),
        (63.0, "Ha, that's a good one. Thank you!"),
    ]
    assert not any(isinstance(step, Say) for step in ben.steps)
    assert scenario.scene.person("mia").voice != scenario.scene.person("ben").voice


@pytest.mark.parametrize(
    ("name", "live", "scripted"),
    [("greeting_as_mia", "mia", "ben"), ("greeting_as_ben", "ben", "mia")],
)
def test_you_speak_for_one_of_them_while_the_other_keeps_their_script(
    name: str, live: str, scripted: str
) -> None:
    scenario = describe(name)

    assert scenario.scripts[live] == Live(idle=scenario.scripts[live].idle, steps=())
    assert scenario.scripts[scripted] == describe("greeting").scripts[scripted]


def test_mia_and_ben_chat_then_meet_the_robot_and_cleo_walks_in() -> None:
    scenario = describe("newcomer")

    lines = sorted(
        (step.at_s, name, step.text)
        for name, script in scenario.scripts.items()
        if isinstance(script, Scripted)
        for step in script.steps
        if isinstance(step, Say)
    )
    assert lines == [
        (2.0, "mia", "So, did you get the demo working in the end?"),
        (6.0, "ben", "Almost. The head still wobbles when it turns."),
        (10.0, "mia", "Look, it's watching us. Let's say hi."),
        (19.5, "mia", "Hi Milo! I'm Mia."),
        (34.0, "ben", "And I'm Ben. Can you tell us a story about a robot who learns to dance?"),
        (50.0, "mia", "Sorry, hold on. Someone's coming."),
        (58.0, "cleo", "Hi, I'm Cleo. I just joined the team."),
        (67.0, "cleo", "Nice to meet you. Who else is here?"),
        (77.0, "ben", "Right, back to work. See you later!"),
    ]
    cleo = scenario.scripts["cleo"]
    assert isinstance(cleo, Scripted)
    assert cleo.steps[0] == Play(
        at_s=41.0,
        clips=(
            ClipName.parse("xy/f_walk_start"),
            ClipName.parse("xy/f_walk_neutral_01"),
            ClipName.parse("xy/f_walk_stop"),
            ClipName.parse("xyz/f_turn_left_60"),
            ClipName.parse("static/f_wave_01"),
        ),
    )


def test_mia_and_ben_face_each_other_and_cleo_starts_out_of_the_robots_view() -> None:
    scene = chatting_by_the_table()

    assert [(p.name, p.turned_deg) for p in scene.people[:2]] == [("mia", 60.0), ("ben", -60.0)]
    x, y = scene.person("cleo").position
    # The head camera sees 40 degrees each side of its axis.
    assert math.degrees(math.atan2(-y, x)) > 70.0
    assert len({person.voice for person in scene.people}) == 3


def test_you_speak_for_mia_who_still_turns_to_the_robot_at_her_time() -> None:
    scenario = describe("newcomer_as_mia")

    assert scenario.scripts["mia"] == Live(
        idle=scenario.scripts["mia"].idle,
        steps=(Play(at_s=13.0, clips=(ClipName.parse("xyz/f_turn_right_60"),)),),
    )
    assert scenario.scripts["ben"] == describe("newcomer").scripts["ben"]
    assert scenario.scripts["cleo"] == describe("newcomer").scripts["cleo"]


def test_a_play_step_plays_at_least_one_clip() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Play(at_s=1.0, clips=())


def test_still_people_stand_silent_in_their_a_pose() -> None:
    assert set(describe("still").scripts.values()) == {Scripted(idle=(), steps=())}


def test_a_live_script_cannot_say_lines() -> None:
    with pytest.raises(ValidationError, match="play"):
        Live.model_validate({"idle": [], "steps": [{"kind": "say", "at_s": 1.0, "text": "Hi"}]})


def test_steps_are_written_in_order() -> None:
    with pytest.raises(ValidationError, match="not in order"):
        Scripted(
            idle=(), steps=(Play(at_s=10.0, clips=(TALK,)), Say(at_s=5.0, text="Hi", gesture=TALK))
        )


def test_everyone_in_the_scene_has_a_script_and_nobody_else() -> None:
    script = Scripted(idle=(), steps=())

    with pytest.raises(ValidationError, match=r"Scripts for \['bob', 'mia'\], not the scene's"):
        Scenario(scene=two_by_the_table(), scripts={"mia": script, "bob": script})


def test_one_microphone_speaks_for_one_person_at_most() -> None:
    live = Live(idle=(), steps=())

    with pytest.raises(ValidationError, match="one microphone speaks for one at most"):
        Scenario(scene=two_by_the_table(), scripts={"mia": live, "ben": live})


def test_names_in_a_scene_are_unique() -> None:
    mia = two_by_the_table().person("mia")

    with pytest.raises(ValidationError, match="each name once"):
        Scene(table_height=1.0, people=(mia, mia))


def test_a_person_is_found_by_name() -> None:
    scene = two_by_the_table()

    assert scene.person("ben") == Person(
        name="ben",
        avatar="Adults/Male_Adult_04",
        voice="am_michael",
        position=(0.8, -0.65),
        turned_deg=0.0,
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
