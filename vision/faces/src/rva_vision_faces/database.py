import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt

from rva_schemas.vision import KnownPerson, UnknownPersonError
from rva_vision_faces.embedder import Embedding

# Users are pseudonymous ids: names live in their own table, and face samples and the operator
# list refer to user ids only.
_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS names (
    user_id TEXT PRIMARY KEY REFERENCES users (user_id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""


_PEOPLE: Final = """
SELECT users.user_id, names.name FROM users JOIN names USING (user_id) ORDER BY users.created_at
"""


class FaceDatabase:
    """Users, their names and samples of their faces in a local SQLite file. No images are stored.

    Each call opens its own connection, so it can run in any thread.
    """

    def __init__(self, path: Path, max_embeddings_per_person: int) -> None:
        self._path: Final = path
        self._max_embeddings: Final = max_embeddings_per_person
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.executescript(_SCHEMA)

    def people(self) -> list[KnownPerson]:
        """Everyone known, oldest first."""
        with self._transaction() as connection:
            rows = connection.execute(_PEOPLE).fetchall()
        return [KnownPerson(user_id=UUID(user_id), name=name) for user_id, name in rows]

    def load(self) -> list[tuple[KnownPerson, npt.NDArray[np.float32]]]:
        """Everyone with at least one sample, with their samples as rows."""
        with self._transaction() as connection:
            loaded = []
            for user_id, name in connection.execute(_PEOPLE).fetchall():
                vectors = connection.execute(
                    "SELECT vector FROM embeddings WHERE user_id = ? ORDER BY id", (user_id,)
                ).fetchall()
                if vectors:
                    samples = np.stack([np.frombuffer(blob, np.float32) for (blob,) in vectors])
                    loaded.append((KnownPerson(user_id=UUID(user_id), name=name), samples))
            return loaded

    def add(self, name: str, embeddings: Sequence[Embedding]) -> KnownPerson:
        """Add samples to the person with this name, creating them if the name is new."""
        now = datetime.now(UTC).isoformat()
        with self._transaction() as connection:
            row = connection.execute("SELECT user_id FROM names WHERE name = ?", (name,)).fetchone()
            if row is None:
                user_id = str(uuid4())
                connection.execute(
                    "INSERT INTO users (user_id, created_at) VALUES (?, ?)", (user_id, now)
                )
                connection.execute(
                    "INSERT INTO names (user_id, name) VALUES (?, ?)", (user_id, name)
                )
            else:
                (user_id,) = row
            connection.executemany(
                "INSERT INTO embeddings (user_id, vector, created_at) VALUES (?, ?, ?)",
                [(user_id, vector.astype(np.float32).tobytes(), now) for vector in embeddings],
            )
            connection.execute(
                """DELETE FROM embeddings WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM embeddings WHERE user_id = ? ORDER BY id DESC LIMIT ?)""",
                (user_id, user_id, self._max_embeddings),
            )
        return KnownPerson(user_id=UUID(user_id), name=name)

    def forget(self, name: str) -> KnownPerson:
        """Delete a user, their name and their samples.

        Raises:
            UnknownPersonError: If nobody has this name.
        """
        with self._transaction() as connection:
            row = connection.execute("SELECT user_id FROM names WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise UnknownPersonError(name)
            connection.execute("DELETE FROM users WHERE user_id = ?", row)
        return KnownPerson(user_id=UUID(row[0]), name=name)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
