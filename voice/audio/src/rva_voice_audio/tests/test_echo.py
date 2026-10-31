from rva_voice_audio.echo import RobotSpeech


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_cut_off_words_of_the_robot_are_its_echo() -> None:
    speech = RobotSpeech(echo_window_s=2, clock=_Clock())
    speech.starts("I'm sorry, I can't reach the shelf.")

    assert speech.is_echo("I'm sor")
    assert speech.is_echo("I can reach the shelf")
    assert not speech.is_echo("Hello, the barge-in is not working.")


def test_echo_is_remembered_for_the_window_after_speaking() -> None:
    clock = _Clock()
    speech = RobotSpeech(echo_window_s=2, clock=clock)
    speech.starts("The cup is on the shelf.")
    clock.now += 10
    speech.stops()

    clock.now += 1.5
    assert speech.is_echo("the cup is on")
    clock.now += 1
    assert not speech.is_echo("the cup is on")
