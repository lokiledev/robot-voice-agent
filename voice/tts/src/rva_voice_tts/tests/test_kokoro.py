import pytest

from rva_voice_tts.kokoro import Respelling, respell

REACHY = (Respelling(written="Reachy", spoken="Reechy"),)


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("Hi! I'm Reachy.", "Hi! I'm Reechy."),
        ("reachy, REACHY's friend", "Reechy, Reechy's friend"),
        # Not inside another word.
        ("Reachyland is far", "Reachyland is far"),
        ("Nothing to change.", "Nothing to change."),
    ],
)
def test_respelled_words_are_replaced_whole(text: str, said: str) -> None:
    assert respell(text, REACHY) == said
