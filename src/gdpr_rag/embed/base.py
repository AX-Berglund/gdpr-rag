"""The embedding interface every backend implements.

Backends are swappable because the evaluation harness compares them: the
question "does a larger embedding model actually retrieve better here?" is
worth answering with numbers rather than assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-norm vectors.

    Implementations must return L2-normalised rows so that a dot product is
    cosine similarity. Normalising once at embedding time keeps it out of the
    search hot path and out of every caller.
    """

    @property
    def name(self) -> str:
        """Identifier recorded alongside results, e.g. in evaluation tables."""
        ...

    @property
    def dimensions(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an array of shape ``(len(texts), dimensions)``."""
        ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving all-zero rows untouched rather than NaN."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
