"""Turning regulation documents into citable chunks."""

from gdpr_rag.ingest.chunk import chunk_document, classify, parse_article, split_articles
from gdpr_rag.ingest.eurlex import parse_document
from gdpr_rag.ingest.fixed import fixed_size_chunks
from gdpr_rag.ingest.models import Chunk, ChunkKind

__all__ = [
    "Chunk",
    "ChunkKind",
    "chunk_document",
    "fixed_size_chunks",
    "classify",
    "parse_article",
    "parse_document",
    "split_articles",
]
