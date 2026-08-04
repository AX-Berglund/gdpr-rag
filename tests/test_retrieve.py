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


class TestTracing:
    def test_retrieval_without_a_trace_is_unchanged(self, retriever):
        # Tracing must be free to ignore.
        assert retriever.retrieve("breach notification", k=1)[0].chunk.article == 33

    def test_a_trace_records_the_query_and_what_came_back(self, retriever):
        from gdpr_rag.trace import Trace

        trace = Trace("breach notification authority")
        retriever.retrieve("breach notification authority", k=2, trace=trace)
        span = trace.find("retrieve")[0]
        assert span.inputs["k"] == 2
        assert span.inputs["embedder"] == "hashing-1024"
        assert "Article 33" in span.outputs["citations"]
        assert len(span.outputs["scores"]) == 2

    def test_repeated_retrievals_appear_as_separate_spans(self, retriever):
        from gdpr_rag.trace import Trace

        trace = Trace("multi")
        retriever.retrieve("erasure", k=1, trace=trace)
        retriever.retrieve("transfers", k=1, trace=trace)
        assert len(trace.find("retrieve")) == 2
