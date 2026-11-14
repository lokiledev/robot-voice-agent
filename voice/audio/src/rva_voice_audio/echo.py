import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Final

_NOT_WORD: Final = re.compile(r"[^\w\s]")
_ECHO_COVERAGE: Final = 0.8
"""Share of a transcript found in the robot's recent speech for it to count as an echo."""


class RobotSpeech:
    """What the robot is saying or just said, shared by voice output and voice input.

    Lets the input tell the user's voice from the robot's own voice coming back through the
    microphone, which echo cancellation does not always remove.
    """

    def __init__(self, echo_window_s: float, clock: Callable[[], float]) -> None:
        """Remember the robot's sentences for `echo_window_s` after it stops speaking."""
        self._echo_window_s: Final = echo_window_s
        self._clock: Final = clock
        self._sentences: list[tuple[float, str]] = []
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def starts(self, sentence: str) -> None:
        self._speaking = True
        self._forget_old()
        self._sentences.append((self._clock(), _normalize(sentence)))

    def stops(self) -> None:
        self._speaking = False
        now = self._clock()
        self._sentences = [(now, sentence) for _, sentence in self._sentences]

    def is_echo(self, transcript: str) -> bool:
        """Whether most of `transcript` is something the robot is saying or just said."""
        self._forget_old()
        heard = _normalize(transcript).split()
        said = " ".join(sentence for _, sentence in self._sentences).split()
        if not heard or not said:
            return False
        # Whole words, never letters inside them: "No" is not the echo of "know". The first
        # catches a word cut off at the end ("I'm sor"), the second a misheard word in the middle
        # ("I can reach" for "I can't reach").
        return _is_cut_off(heard, said) or _matched_share(heard, said) >= _ECHO_COVERAGE

    def _forget_old(self) -> None:
        if self._speaking:
            return
        oldest = self._clock() - self._echo_window_s
        self._sentences = [(at, sentence) for at, sentence in self._sentences if at >= oldest]


def _normalize(text: str) -> str:
    return " ".join(_NOT_WORD.sub("", text.lower()).split())


def _is_cut_off(heard: list[str], said: list[str]) -> bool:
    """Whether `heard` is a run of the words of `said`, its last word cut short.

    Never for a single word: "Stop" is not the echo of "stopped".
    """
    *whole, cut = heard
    if not whole:
        return False
    return any(
        said[start : start + len(whole)] == whole and said[start + len(whole)].startswith(cut)
        for start in range(len(said) - len(whole))
    )


def _matched_share(heard: list[str], said: list[str]) -> float:
    blocks = SequenceMatcher(None, heard, said, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks) / len(heard)
