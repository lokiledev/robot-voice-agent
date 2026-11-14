import pydantic
import pytest

from rva_schemas.profile import Capabilities


def _capabilities(greeting_animation: str | None) -> Capabilities:
    return Capabilities(
        body="a small desktop robot",
        skills=(),
        walks=False,
        looks_at_people=False,
        animations=("wiggle_antennas", "nod"),
        greeting_animation=greeting_animation,
        constraints=(),
    )


def test_a_robot_greets_with_one_of_its_animations_or_none() -> None:
    assert _capabilities("wiggle_antennas").greeting_animation == "wiggle_antennas"
    assert _capabilities(None).greeting_animation is None
    with pytest.raises(pydantic.ValidationError, match="wave_right_hand is not one of"):
        _capabilities("wave_right_hand")
