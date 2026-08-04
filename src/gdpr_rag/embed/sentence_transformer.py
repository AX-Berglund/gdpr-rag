"""Local dense embeddings via sentence-transformers.

This is the default backend: it runs on Apple Silicon through MPS, needs no API
key and no network at query time, and keeps the whole project runnable from a
clean checkout with nothing but a model download.

The import is deferred so that ``gdpr_rag`` stays importable -- and testable --
without torch installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gdpr_rag.embed.base import normalise

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_device(requested: str | None) -> str:
    """Pick the best available device, preferring MPS on Apple Silicon."""
    import torch

    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    """Dense embeddings from a local sentence-transformers checkpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._requested_device = device
        self._batch_size = batch_size
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Normalising here rather than trusting the model's own flag keeps the
        # unit-norm guarantee true for every checkpoint.
        return normalise(vectors)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Local embeddings need the 'local' extra: pip install 'gdpr-rag[local]'"
                ) from exc
            self._model = SentenceTransformer(
                self._model_name, device=_resolve_device(self._requested_device)
            )
        return self._model
