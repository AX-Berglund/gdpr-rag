"""Persistent chunk storage with brute-force cosine search.

An approximate index (FAISS, HNSW, pgvector) buys nothing at this scale: the
GDPR is on the order of a thousand chunks, so an exact dot product against the
whole matrix is well under a millisecond and always returns the true top-k.
Choosing brute force here means the retrieval numbers in the evaluation are not
confounded by index recall.

SQLite holds the chunks and their vectors; the matrix is loaded once into
memory on first search. Swapping in pgvector later is a matter of implementing
the same two methods.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gdpr_rag.embed.base import normalise
from gdpr_rag.ingest.models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    citation   TEXT NOT NULL,
    payload    TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity to the query."""

    chunk: Chunk
    score: float

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.chunk.citation}"


class ChunkStore:
    """Stores chunks with their embeddings and searches them by cosine similarity."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        # A store is built once and then queried from wherever queries arrive,
        # which for any served application means a different thread than the
        # one that built it. sqlite3 refuses that by default, so cross-thread
        # use is enabled explicitly and serialised with a lock rather than
        # left to chance. The lock is reentrant because search() holds it
        # while _matrix() takes it again.
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.executescript(_SCHEMA)
        self._cache: tuple[np.ndarray, list[Chunk]] | None = None

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Store ``chunks`` alongside their embeddings.

        Embeddings are normalised on the way in so that search is a plain dot
        product regardless of which backend produced them.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(embeddings)} embeddings; they must match"
            )
        if not chunks:
            return

        embeddings = normalise(embeddings)
        with self._lock:
            self._connection.executemany(
                "INSERT INTO chunks (citation, payload, embedding) VALUES (?, ?, ?)",
                [
                    (chunk.citation, chunk.model_dump_json(), vector.astype(np.float32).tobytes())
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )
            self._connection.commit()
            self._cache = None

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` most similar chunks, highest score first."""
        if k < 1:
            raise ValueError("k must be at least 1")

        with self._lock:
            matrix, chunks = self._matrix()
        if not chunks:
            return []

        query = normalise(query_embedding)[0]
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions but the store holds {matrix.shape[1]}; "
                "the query and the index must come from the same embedder"
            )

        scores = matrix @ query
        # argpartition finds the top-k without sorting the whole corpus, then
        # only those k are sorted.
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in top]

    def set_meta(self, key: str, value: str) -> None:
        """Record provenance, e.g. which embedder built this index."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._connection.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __len__(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def __enter__(self) -> ChunkStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _matrix(self) -> tuple[np.ndarray, list[Chunk]]:
        """Load and cache the embedding matrix. Caller must hold the lock."""
        if self._cache is None:
            rows = self._connection.execute(
                "SELECT payload, embedding FROM chunks ORDER BY id"
            ).fetchall()
            chunks = [Chunk(**json.loads(payload)) for payload, _ in rows]
            matrix = (
                np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
                if rows
                else np.zeros((0, 0), dtype=np.float32)
            )
            self._cache = (matrix, chunks)
        return self._cache
