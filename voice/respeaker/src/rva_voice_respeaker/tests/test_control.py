import pytest

from rva_voice_respeaker.control import ReSpeakerUnavailableError, parse_floats


def test_a_reply_is_its_status_then_little_endian_floats() -> None:
    reply = bytes([0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xBF])

    assert parse_floats(reply, 2) == (1.0, -0.5)


def test_a_reply_not_ready_asks_to_be_asked_again() -> None:
    assert parse_floats(bytes([0x40, 0x00, 0x00, 0x00, 0x00]), 1) is None


def test_any_other_status_is_an_error() -> None:
    with pytest.raises(ReSpeakerUnavailableError):
        parse_floats(bytes([0x01, 0x00, 0x00, 0x00, 0x00]), 1)
