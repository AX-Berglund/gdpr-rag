"""Tests for the retriever."""

import pytest

from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.ingest.models import Chunk, ChunkKind
from gdpr_rag.retrieve import Retriever

CHUNKS = [
    Chunk(article=17, kind=ChunkKind.BODY, text="the right to erasure of personal data"),
    Chunk(article=33, kind=ChunkKind.BODY, text="notification of a breach to the authority"),
    Chunk(article=44, kind=ChunkKind.BODY, text="transfers of data to third countries"),
]


@pytest.fixture
def retriever():
    r = Retriever.build(HashingEmbedder(dimensions=1024), CHUNKS)
    yield r
    r.store.close()


def test_retrieves_the_relevant_article(retriever):
    assert retriever.retrieve("breach notification authority", k=1)[0].chunk.article == 33


def test_respects_k(retriever):
    assert len(retriever.retrieve("personal data", k=2)) == 2


def test_build_records_the_embedder_that_made_the_index(retriever):
    assert retriever.store.get_meta("embedder") == "hashing-1024"


def test_empty_question_is_rejected(retriever):
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.retrieve("  ")


def test_building_over_no_chunks_returns_nothing():
    r = Retriever.build(HashingEmbedder(dimensions=64), [])
    assert r.retrieve("anything") == []
    r.store.close()
