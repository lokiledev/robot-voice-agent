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
        heard = _normalize(transcript)
        said = " ".join(sentence for _, sentence in self._sentences)
        if not heard or not said:
            return False
        # Characters catch a word cut off at the end ("I'm sor"); words catch a misheard word in
        # the middle ("I can reach" for "I can't reach").
        return (
            _longest_match_share(heard, said) >= _ECHO_COVERAGE
            or _matched_share(heard.split(), said.split()) >= _ECHO_COVERAGE
        )

    def _forget_old(self) -> None:
        if self._speaking:
            return
        oldest = self._clock() - self._echo_window_s
        self._sentences = [(at, sentence) for at, sentence in self._sentences if at >= oldest]


def _normalize(text: str) -> str:
    return " ".join(_NOT_WORD.sub("", text.lower()).split())


def _longest_match_share(heard: str, said: str) -> float:
    return SequenceMatcher(None, heard, said, autojunk=False).find_longest_match().size / len(heard)


def _matched_share(heard: list[str], said: list[str]) -> float:
    blocks = SequenceMatcher(None, heard, said, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks) / len(heard)
