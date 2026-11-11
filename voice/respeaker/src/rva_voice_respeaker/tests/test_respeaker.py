import math

from rva_voice_respeaker.respeaker import speaker_azimuth

# Read from the board while someone talked at 176 degrees, then while nobody did.
TALKING_AZIMUTHS = (
    math.radians(161.0),
    math.radians(339.0),
    math.radians(175.0),
    math.radians(175.0),
)


def test_the_speaker_is_where_the_focused_beam_hearing_the_most_speech_points() -> None:
    energies = (1.43e6, 0.0, 2.35e6, 2.35e6)

    assert speaker_azimuth(TALKING_AZIMUTHS, energies) == math.radians(161.0)


def test_the_other_focused_beam_counts_when_it_hears_more() -> None:
    energies = (2.0e5, 9.0e5, 0.0, 0.0)

    assert speaker_azimuth(TALKING_AZIMUTHS, energies) == math.radians(339.0)


def test_nobody_speaks_while_the_focused_beams_hear_no_speech() -> None:
    # The scanning and selected beams' energies are left out: they follow the focused ones.
    energies = (0.0, 0.0, 3.0e5, 3.0e5)

    assert speaker_azimuth(TALKING_AZIMUTHS, energies) is None
