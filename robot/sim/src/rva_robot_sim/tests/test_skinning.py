import math

import numpy as np
import pytest

from rva_robot_sim.skinning import (
    Clip,
    Skin,
    deform,
    displacement,
    nlerp,
    placement,
    pose,
    retarget,
    sample,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)
QUARTER_TURN_Z = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
"""A quarter turn counterclockwise about +z, as w, x, y, z."""


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def _skin() -> Skin:
    """A hip 1 m up and a chest 1 m above it; one vertex on each, and one on both."""
    return Skin(
        bone_names=("hip", "chest"),
        bone_parents=np.array([-1, 0], dtype=np.int32),
        bone_rest=np.stack([_translation(0, 0, 1), _translation(0, 0, 2)]),
        positions=np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0, 1.5]]),
        bone_index=np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
        bone_weight=np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0.5, 0.5, 0, 0]]),
        shape_names=("raise",),
        shape_deltas=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]]),
    )


def test_the_rest_pose_leaves_the_mesh_as_it_is() -> None:
    skin = _skin()
    bones = pose(skin, np.array([IDENTITY, IDENTITY]), np.array([0.0, 0.0, 1.0]), np.eye(4))

    np.testing.assert_allclose(bones, skin.bone_rest)
    np.testing.assert_allclose(deform(skin, bones, np.array([0.0])), skin.positions)


def test_a_turned_bone_carries_its_vertices_and_half_of_those_it_shares() -> None:
    skin = _skin()
    bones = pose(skin, np.array([IDENTITY, QUARTER_TURN_Z]), np.array([0.0, 0.0, 1.0]), np.eye(4))

    np.testing.assert_allclose(
        deform(skin, bones, np.array([0.0])),
        # Blended linearly: halfway between where each bone would put it.
        [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.5, 0.5, 1.5]],
        atol=1e-12,
    )


def test_face_shapes_move_the_rest_pose_before_skinning() -> None:
    skin = _skin()
    bones = pose(skin, np.array([IDENTITY, QUARTER_TURN_Z]), np.array([0.0, 0.0, 1.0]), np.eye(4))

    np.testing.assert_allclose(deform(skin, bones, np.array([0.5]))[1], [0.0, 1.0, 2.1], atol=1e-12)


def test_a_placement_moves_and_turns_the_whole_avatar() -> None:
    skin = _skin()
    bones = pose(
        skin,
        np.array([IDENTITY, IDENTITY]),
        np.array([0.0, 0.0, 1.0]),
        placement(2.0, 0.0, math.pi / 2),
    )

    np.testing.assert_allclose(deform(skin, bones, np.array([0.0]))[0], [2.0, 1.0, 1.0], atol=1e-12)


def _clip(root: tuple[float, ...], chest: tuple[float, ...], heights: tuple[float, ...]) -> Clip:
    """Two frames from a skeleton half the skin's height, with an arm the skin has not."""
    return Clip(
        bone_names=("chest", "arm", "hip"),
        rotations=np.array([[chest, IDENTITY, root], [chest, IDENTITY, root]]),
        root_position=np.array([[3.0, 4.0, heights[0]], [3.5, 4.0, heights[1]]]),
        rest_root_height=0.5,
        fps=10.0,
    )


def test_retargeting_gives_the_clips_rotations_to_the_skins_bones_by_name() -> None:
    motion = retarget(_skin(), _clip(IDENTITY, QUARTER_TURN_Z, (0.5, 0.4)), still=frozenset())

    np.testing.assert_allclose(motion.rotations[0], [IDENTITY, QUARTER_TURN_Z])
    # Scaled to the skin's hip height, from above where the skin's hip is at rest.
    np.testing.assert_allclose(motion.root_position, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.8]])


def test_retargeting_keeps_still_bones_at_rest() -> None:
    motion = retarget(
        _skin(), _clip(IDENTITY, QUARTER_TURN_Z, (0.5, 0.5)), still=frozenset({"chest"})
    )

    np.testing.assert_allclose(motion.rotations[:, 1], [IDENTITY, IDENTITY])


def test_a_motion_starts_facing_as_the_skin_does_at_rest() -> None:
    motion = retarget(_skin(), _clip(QUARTER_TURN_Z, IDENTITY, (0.5, 0.5)), still=frozenset())

    np.testing.assert_allclose(motion.rotations[:, 0], [IDENTITY, IDENTITY], atol=1e-12)
    # The clip's step along +x, turned with it: along -y once facing as the skin.
    np.testing.assert_allclose(motion.root_position[1], [0.0, -1.0, 1.0], atol=1e-12)


def test_retargeting_needs_every_bone_that_is_not_still() -> None:
    skin = _skin().model_copy(update={"bone_names": ("hip", "tail")})

    with pytest.raises(ValueError, match="tail"):
        retarget(skin, _clip(IDENTITY, IDENTITY, (0.5, 0.5)), still=frozenset())


def test_sampling_interpolates_between_frames_and_clamps_to_the_motion() -> None:
    motion = retarget(_skin(), _clip(IDENTITY, IDENTITY, (0.5, 0.4)), still=frozenset())

    _, halfway = sample(motion, 0.05)
    _, after = sample(motion, 3.0)

    np.testing.assert_allclose(halfway, [0.5, 0.0, 0.9])
    np.testing.assert_allclose(after, [1.0, 0.0, 0.8])


def test_displacement_is_how_far_the_motion_went_and_turned() -> None:
    skin = _skin()
    motion = retarget(skin, _clip(IDENTITY, IDENTITY, (0.5, 0.5)), still=frozenset())
    turned = motion.model_copy(
        update={"rotations": np.array([[IDENTITY, IDENTITY], [QUARTER_TURN_Z, IDENTITY]])}
    )

    np.testing.assert_allclose(
        displacement(skin, turned, turned.duration), placement(1.0, 0.0, math.pi / 2), atol=1e-12
    )


def test_nlerp_takes_the_shorter_way() -> None:
    np.testing.assert_allclose(nlerp(np.array(IDENTITY), -np.array(IDENTITY), 0.5), IDENTITY)
