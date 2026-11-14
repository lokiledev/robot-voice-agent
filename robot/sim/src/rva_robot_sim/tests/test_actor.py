import math

import numpy as np
import pytest

from rva_robot_sim.actor import Actor, vertex_order
from rva_robot_sim.animation_assets import ClipName
from rva_robot_sim.skinning import Motion, Skin, placement

IDENTITY = (1.0, 0.0, 0.0, 0.0)
QUARTER_TURN_Z = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def _skin() -> Skin:
    """A hip 1 m up, a chest 1 m above it with a mouth vertex, and the face shapes an actor uses."""
    zero = np.zeros((2, 3))
    mouth = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.02]])
    return Skin(
        bone_names=("hip", "chest"),
        bone_parents=np.array([-1, 0], dtype=np.int32),
        bone_rest=np.stack([_translation(0, 0, 1), _translation(0, 0, 2)]),
        positions=np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 2.0]]),
        bone_index=np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int32),
        bone_weight=np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        shape_names=("AK_09_EyeBlinkLeft", "AK_10_EyeBlinkRight", "AA_VI_10_aa"),
        shape_deltas=np.stack([zero, zero, mouth]),
    )


def _motion(root_rotations: tuple[tuple[float, ...], ...]) -> Motion:
    """One frame a second, the hip turning as given, standing where the skin's hip is at rest."""
    return Motion(
        rotations=np.array([[rotation, IDENTITY] for rotation in root_rotations]),
        root_position=np.array([[0.0, 0.0, 1.0]] * len(root_rotations)),
        fps=1.0,
    )


def _clip(text: str) -> ClipName:
    return ClipName.parse(text)


def _shut(time: float) -> float:
    return 0.0


def test_without_clips_they_stand_still_where_placed() -> None:
    skin = _skin()
    actor = Actor(skin, [], placement(2.0, 0.0, 0.0), _shut)

    for time in (0.0, 10.0, 1000.0):
        np.testing.assert_allclose(actor.pose(time), placement(2.0, 0.0, 0.0) @ skin.bone_rest)


def test_clips_play_one_after_the_other_and_loop() -> None:
    still = _motion((IDENTITY, IDENTITY))
    actor = Actor(
        _skin(),
        [(_clip("static/f_idle_neutral_01"), still), (_clip("static/f_idle_breathe_01"), still)],
        placement(0.0, 0.0, 0.0),
        _shut,
    )

    actor.pose(1.5)
    actor.pose(4.5)

    np.testing.assert_allclose(actor.anchor, placement(0.0, 0.0, 0.0), atol=1e-12)


def test_a_turn_leaves_the_next_clip_facing_the_new_way() -> None:
    turn = _motion((IDENTITY, QUARTER_TURN_Z))
    still = _motion((IDENTITY, IDENTITY))
    actor = Actor(
        _skin(),
        [(_clip("xyz/f_turn_left_90"), turn), (_clip("static/f_idle_neutral_01"), still)],
        placement(0.0, 0.0, 0.0),
        _shut,
    )

    bones = actor.pose(1.5)

    np.testing.assert_allclose(actor.anchor, placement(0.0, 0.0, math.pi / 2), atol=1e-12)
    np.testing.assert_allclose(bones[0, :3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_placing_them_moves_the_current_clip() -> None:
    actor = Actor(_skin(), [], placement(0.0, 0.0, 0.0), _shut)

    actor.place(placement(0.0, 3.0, 0.0))

    np.testing.assert_allclose(actor.pose(0.0)[0, :3, 3], [0.0, 3.0, 1.0])


def test_time_does_not_go_back() -> None:
    still = _motion((IDENTITY, IDENTITY))
    actor = Actor(_skin(), [(_clip("static/f_idle_neutral_01"), still)], placement(0, 0, 0), _shut)
    actor.pose(2.5)

    with pytest.raises(ValueError, match="went back"):
        actor.pose(0.5)


@pytest.mark.parametrize(("opening", "opens"), [(0.0, False), (1.0, True)])
def test_the_mouth_opens_as_its_speech_says(opening: float, opens: bool) -> None:
    skin = _skin()
    actor = Actor(skin, [], placement(0, 0, 0), lambda time: opening)

    mouth = actor.vertices(actor.pose(1.0), 1.0)[1]

    assert (float(mouth[2]) < 2.0) is opens


def test_genesis_vertices_find_the_skin_vertex_at_their_place() -> None:
    skin = _skin()
    # Split at a seam: the chest's vertex twice.
    loaded = np.array([[0.1, 0.0, 2.0], [0.1, 0.0, 1.0], [0.1, 0.0, 2.0]])

    np.testing.assert_array_equal(vertex_order(skin, loaded), [1, 0, 1])


def test_a_vertex_the_skin_has_not_is_an_error() -> None:
    with pytest.raises(ValueError, match="from the skin's"):
        vertex_order(_skin(), np.array([[0.5, 0.0, 2.0]]))
