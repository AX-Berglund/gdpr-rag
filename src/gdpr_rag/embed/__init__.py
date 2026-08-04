"""Embedding backends."""

from gdpr_rag.embed.base import Embedder, normalise
from gdpr_rag.embed.hashing import HashingEmbedder
from gdpr_rag.embed.sentence_transformer import DEFAULT_MODEL, SentenceTransformerEmbedder

__all__ = [
    "DEFAULT_MODEL",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "normalise",
]
