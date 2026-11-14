from pathlib import Path

import pytest
from pydantic import ValidationError

from rva_pydantic import load_yaml
from rva_robot_sim.config import SimConfig
from rva_robot_sim.scenes import SCENES, Line, Person, describe

SIM_YAML = Path(__file__).parents[5] / "config" / "reachy_mini_sim" / "sim.yaml"


@pytest.mark.parametrize("name", sorted(SCENES))
def test_every_scene_describes_its_people(name: str) -> None:
    scene = describe(name)

    assert scene.people
    assert all(person.avatar.startswith("Adults/") for person in scene.people)


def test_the_pair_is_still_or_animated_in_the_same_places() -> None:
    still, animated = describe("still_pair"), describe("animated_pair")

    assert [person.position for person in still.people] == [(1.1, 0.0), (0.8, -0.65)]
    assert [person.position for person in animated.people] == [(1.1, 0.0), (0.8, -0.65)]
    assert all(not person.clips for person in still.people)
    assert all(person.clips for person in animated.people)
    assert all(not person.lines for person in still.people)
    # She talks to the robot; he only listens.
    her, him = animated.people
    assert [line.at_s for line in her.lines] == [3.0, 23.0, 43.0, 63.0]
    assert not him.lines
    assert her.voice != him.voice


def test_lines_are_said_in_order() -> None:
    person = describe("animated_pair").people[0]
    later, sooner = Line(at_s=10.0, text="Later."), Line(at_s=5.0, text="Sooner.")

    with pytest.raises(ValidationError, match="not in order"):
        Person.model_validate({**person.model_dump(), "lines": [later, sooner]})


def test_an_unknown_scene_lists_the_others() -> None:
    with pytest.raises(KeyError, match="animated_pair"):
        describe("crowd")


def test_the_reachy_profile_runs_the_animated_pair() -> None:
    config = load_yaml(SimConfig, SIM_YAML)

    assert config.scene == "animated_pair"


def test_a_profile_cannot_name_a_scene_that_does_not_exist(tmp_path: Path) -> None:
    text = SIM_YAML.read_text()
    assert "scene: animated_pair\n" in text
    misnamed = tmp_path / "sim.yaml"
    misnamed.write_text(text.replace("scene: animated_pair\n", "scene: crowd\n"))

    with pytest.raises(ValidationError, match="No scene 'crowd'"):
        load_yaml(SimConfig, misnamed)
