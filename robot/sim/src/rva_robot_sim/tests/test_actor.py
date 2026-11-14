import math

import numpy as np
import pytest

from rva_robot_sim.actor import Action, Actor, saying, vertex_order
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


STILL_2S = _motion((IDENTITY, IDENTITY, IDENTITY))
TURN_1S = _motion((IDENTITY, QUARTER_TURN_Z))
IDLE = ClipName.parse("static/f_idle_neutral_01")
TURN = ClipName.parse("xyz/f_turn_left_90")


def _shut(time: float) -> float:
    return 0.0


def test_without_clips_they_stand_still_where_placed() -> None:
    skin = _skin()
    actor = Actor(skin, [], [], placement(2.0, 0.0, 0.0), _shut)

    for time in (0.0, 10.0, 1000.0):
        np.testing.assert_allclose(actor.pose(time), placement(2.0, 0.0, 0.0) @ skin.bone_rest)


def test_idle_clips_play_one_after_the_other_and_loop() -> None:
    still = _motion((IDENTITY, IDENTITY))
    actor = Actor(_skin(), [(IDLE, still), (IDLE, still)], [], placement(0, 0, 0), _shut)

    actor.pose(1.5)
    actor.pose(4.5)

    np.testing.assert_allclose(actor.anchor, placement(0.0, 0.0, 0.0), atol=1e-12)


def test_a_turn_leaves_the_next_clip_facing_the_new_way() -> None:
    actor = Actor(_skin(), [(TURN, TURN_1S), (IDLE, STILL_2S)], [], placement(0, 0, 0), _shut)

    bones = actor.pose(1.5)

    np.testing.assert_allclose(actor.anchor, placement(0.0, 0.0, math.pi / 2), atol=1e-12)
    np.testing.assert_allclose(bones[0, :3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_an_action_cuts_the_idle_clip_then_idling_goes_on() -> None:
    turning = Action(at_s=1.0, clips=((TURN, TURN_1S),))
    actor = Actor(_skin(), [(IDLE, STILL_2S)], [turning], placement(0, 0, 0), _shut)
    actor.begin(0.0)

    actor.pose(0.5)
    idle_anchor = actor.anchor.copy()
    actor.pose(1.5)
    acting_anchor = actor.anchor.copy()
    bones = actor.pose(2.5)

    np.testing.assert_allclose(idle_anchor, placement(0, 0, 0), atol=1e-12)
    # Cut a second into the idle clip, which does not move them.
    np.testing.assert_allclose(acting_anchor, placement(0, 0, 0), atol=1e-12)
    # Idle again from 2 s, facing where the turn left them.
    np.testing.assert_allclose(actor.anchor, placement(0, 0, math.pi / 2), atol=1e-12)
    np.testing.assert_allclose(bones[0, :3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_an_action_starting_before_the_last_ends_is_refused() -> None:
    first = Action(at_s=1.0, clips=((TURN, TURN_1S), (TURN, TURN_1S)))
    second = Action(at_s=2.5, clips=((TURN, TURN_1S),))

    with pytest.raises(ValueError, match="step at 2.5 s starts before the one at 1 s ends, at 3"):
        Actor(_skin(), [], [first, second], placement(0, 0, 0), _shut)


@pytest.mark.parametrize(
    ("line_s", "durations"),
    [
        (0.45, [0.4]),  # Cut at its last frame before the line's end.
        (1.0, [1.0]),
        (2.5, [1.0, 1.0, 0.5]),
    ],
)
def test_a_gesture_lasts_as_long_as_the_line(line_s: float, durations: list[float]) -> None:
    # A second long, at 10 frames a second.
    gesture = Motion(
        rotations=np.array([[IDENTITY, IDENTITY]] * 11),
        root_position=np.array([[0.0, 0.0, 1.0]] * 11),
        fps=10.0,
    )

    action = saying(3.0, line_s, TURN, gesture)

    assert [motion.duration for _, motion in action.clips] == pytest.approx(durations)
    assert action.at_s == 3.0


def test_placing_them_moves_the_current_clip() -> None:
    actor = Actor(_skin(), [], [], placement(0.0, 0.0, 0.0), _shut)

    actor.place(placement(0.0, 3.0, 0.0))

    np.testing.assert_allclose(actor.pose(0.0)[0, :3, 3], [0.0, 3.0, 1.0])


def test_time_does_not_go_back() -> None:
    actor = Actor(_skin(), [(IDLE, _motion((IDENTITY, IDENTITY)))], [], placement(0, 0, 0), _shut)
    actor.pose(2.5)

    with pytest.raises(ValueError, match="went back"):
        actor.pose(0.5)


def test_actions_wait_for_the_script_to_begin_then_count_from_it() -> None:
    turning = Action(at_s=1.0, clips=((TURN, TURN_1S),))
    actor = Actor(_skin(), [(IDLE, STILL_2S)], [turning], placement(0, 0, 0), _shut)

    actor.pose(9.5)
    idling = actor.anchor.copy()
    actor.begin(10.0)
    actor.pose(10.5)
    before_its_time = actor.anchor.copy()
    actor.pose(12.5)

    np.testing.assert_allclose(idling, placement(0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(before_its_time, placement(0, 0, 0), atol=1e-12)
    # Turned from 11 s to 12 s.
    np.testing.assert_allclose(actor.anchor, placement(0, 0, math.pi / 2), atol=1e-12)


@pytest.mark.parametrize(("opening", "opens"), [(0.0, False), (1.0, True)])
def test_the_mouth_opens_as_its_speech_says(opening: float, opens: bool) -> None:
    skin = _skin()
    times: list[float] = []

    def mouth(time: float) -> float:
        times.append(time)
        return opening

    actor = Actor(skin, [], [], placement(0, 0, 0), mouth)
    actor.begin(2.0)

    mouth_vertex = actor.vertices(actor.pose(3.0), 3.0)[1]

    assert (float(mouth_vertex[2]) < 2.0) is opens
    # Asked for the time since the script began.
    assert times == [1.0]


def test_the_mouth_stays_shut_until_the_script_begins() -> None:
    actor = Actor(_skin(), [], [], placement(0, 0, 0), lambda time: 1.0)

    mouth = actor.vertices(actor.pose(3.0), 3.0)[1]

    assert float(mouth[2]) == 2.0


def test_genesis_vertices_find_the_skin_vertex_at_their_place() -> None:
    skin = _skin()
    # Split at a seam: the chest's vertex twice.
    loaded = np.array([[0.1, 0.0, 2.0], [0.1, 0.0, 1.0], [0.1, 0.0, 2.0]])

    np.testing.assert_array_equal(vertex_order(skin, loaded), [1, 0, 1])


def test_a_vertex_the_skin_has_not_is_an_error() -> None:
    with pytest.raises(ValueError, match="from the skin's"):
        vertex_order(_skin(), np.array([[0.5, 0.0, 2.0]]))
