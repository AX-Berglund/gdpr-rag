"""A dependency-free lexical embedder.

This exists for two reasons. It is the floor the evaluation harness measures
against -- a dense model that cannot beat hashed word counts on this corpus is
not earning its download. And it lets the whole pipeline be tested without
pulling in torch, which keeps CI fast.

It is deliberately crude: hashed unigrams with sublinear term weighting. No
IDF, no subword handling, no semantics.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from gdpr_rag.embed.base import normalise

_TOKEN = re.compile(r"[a-z0-9']+")


class HashingEmbedder:
    """Hashes tokens into a fixed-width vector of sublinear term counts."""

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token, count in Counter(_TOKEN.findall(text.lower())).items():
                # Sublinear weighting: a term appearing ten times is more
                # relevant than one appearing once, but not ten times more.
                vectors[row, self._bucket(token)] += 1.0 + math.log(count)
        return normalise(vectors)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimensions
