import numpy as np

from rva_robot_sim.room import Mix


def test_clips_play_together_until_each_ends() -> None:
    mix = Mix()
    mix.start(np.array([1.0, 1.0, 1.0, 1.0], np.float32))
    mix.start(np.array([0.5, 0.5], np.float32))

    played = [mix.next(3).tolist() for _ in range(3)]

    assert played == [[1.5, 1.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_a_clip_stopped_plays_no_further_and_the_others_go_on() -> None:
    mix = Mix()
    cut = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    mix.start(cut)
    mix.start(np.array([0.5, 0.5, 0.5, 0.5], np.float32))

    before = mix.next(2).tolist()
    mix.stop(cut)

    assert (before, mix.next(2).tolist()) == ([1.5, 1.5], [0.5, 0.5])
