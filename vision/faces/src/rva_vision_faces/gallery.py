from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from rva_pydantic import RVABaseModel
from rva_schemas.vision import KnownPerson
from rva_vision_faces.config import MatchConfig
from rva_vision_faces.embedder import Embedding


class Match(RVABaseModel):
    person: KnownPerson
    score: float
    """Best cosine similarity between the face's samples and the person's."""


class Gallery:
    """The known people's face samples, matched in memory. Immutable: rebuilt on every change."""

    def __init__(
        self, people: Sequence[tuple[KnownPerson, npt.NDArray[np.float32]]], config: MatchConfig
    ) -> None:
        """Build the gallery.

        Args:
            people: Each person with their samples, one unit embedding per row.
            config: Recognition thresholds.
        """
        self._people: Final = tuple(person for person, _ in people)
        self._samples: Final = (
            np.concatenate([samples for _, samples in people])
            if people
            else np.zeros((0, 0), np.float32)
        )
        self._owners: Final = np.concatenate(
            [np.full(len(samples), index) for index, (_, samples) in enumerate(people)]
            or [np.zeros(0, np.int64)]
        )
        self._config: Final = config

    def match(self, embeddings: Sequence[Embedding]) -> Match | None:
        """The person these samples of one face belong to, if any is clearly the one."""
        if not self._people or not embeddings:
            return None
        similarities = (np.stack(embeddings) @ self._samples.T).max(axis=0)
        best_per_person = np.full(len(self._people), -1.0)
        np.maximum.at(best_per_person, self._owners, similarities)
        ranked = np.argsort(best_per_person)[::-1]
        best = float(best_per_person[ranked[0]])
        if best < self._config.threshold:
            return None
        if len(ranked) > 1 and best - best_per_person[ranked[1]] < self._config.margin:
            return None
        return Match(person=self._people[ranked[0]], score=best)

    def assign(self, faces: Mapping[int, Sequence[Embedding]]) -> dict[int, Match | None]:
        """Match several faces in view at once: each person goes to at most one, the best scoring."""
        matches = {key: self.match(embeddings) for key, embeddings in faces.items()}
        best_face: dict[KnownPerson, int] = {}
        for key, match in matches.items():
            if match is None:
                continue
            other = best_face.get(match.person)
            if other is None or match.score > _score(matches[other]):
                best_face[match.person] = key
        return {
            key: match if match is not None and best_face[match.person] == key else None
            for key, match in matches.items()
        }


def _score(match: Match | None) -> float:
    assert match is not None
    return match.score
