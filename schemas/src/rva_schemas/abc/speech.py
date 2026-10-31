from collections.abc import AsyncIterator
from typing import Protocol

from rva_schemas.audio import AudioClip


class SpeechRecognizer(Protocol):
    async def transcribe(self, clip: AudioClip) -> str:
        """Transcribe one complete utterance."""
        ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioClip]:
        """Synthesize one sentence, yielding audio as soon as it is ready."""
        ...
