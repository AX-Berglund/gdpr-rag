"""Turning regulation documents into citable chunks."""

from gdpr_rag.ingest.chunk import chunk_document, classify, parse_article, split_articles
from gdpr_rag.ingest.models import Chunk, ChunkKind

__all__ = [
    "Chunk",
    "ChunkKind",
    "chunk_document",
    "classify",
    "parse_article",
    "split_articles",
]
