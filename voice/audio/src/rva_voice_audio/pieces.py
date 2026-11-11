from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_schemas.dialogue import SpeechSpan
from rva_voice_audio.vad import SAMPLE_RATE

_HOP: Final = SAMPLE_RATE // 50
"""20 ms: the stretch whose loudness is compared, to cut between words."""
_REACH: Final = SAMPLE_RATE // 4
"""How far from where it was asked a cut may move, either way, to fall between words."""

type Audio = npt.NDArray[np.float32]


def cut(
    audio: Audio, speech: SpeechSpan, pieces: Sequence[SpeechSpan]
) -> list[tuple[SpeechSpan, Audio]]:
    """A turn's audio cut into `pieces`, each with the span it ends up covering.

    `audio` starts at `speech.start`; `pieces` cover `speech` in order. Where one piece ends and
    the next starts is known to a fraction of a second only, so each cut moves to the quietest
    moment nearby, between two words. The last piece keeps the silence that ended the turn.
    """
    cuts = [
        _quietest(audio, round((piece.start - speech.start) * SAMPLE_RATE)) for piece in pieces[1:]
    ]
    starts, ends = [0, *cuts], [*cuts, len(audio)]
    times = [speech.start, *(speech.start + at / SAMPLE_RATE for at in cuts), speech.end]
    return [
        (SpeechSpan(start=times[i], end=times[i + 1]), audio[starts[i] : ends[i]])
        for i in range(len(pieces))
    ]


def _quietest(audio: Audio, around: int) -> int:
    """The middle of the quietest 20 ms within reach of the sample `around`."""
    first = max(around - _REACH, 0)
    hops = (min(around + _REACH, len(audio)) - first) // _HOP
    if hops <= 0:
        return min(max(around, 0), len(audio))
    window = audio[first : first + hops * _HOP].reshape(hops, _HOP)
    return first + int(np.argmin(np.square(window).sum(axis=1))) * _HOP + _HOP // 2
