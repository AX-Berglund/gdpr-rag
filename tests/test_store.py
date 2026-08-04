"""Tests for chunk storage and cosine search."""

import numpy as np
import pytest

from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.ingest.models import Chunk, ChunkKind
from gdpr_rag.store import ChunkStore


def make_chunk(article: int, text: str, paragraph: str | None = None) -> Chunk:
    return Chunk(
        article=article,
        kind=ChunkKind.PARAGRAPH if paragraph else ChunkKind.BODY,
        text=text,
        paragraph=paragraph,
    )


CORPUS = [
    make_chunk(17, "the right to obtain erasure of personal data without undue delay", "1"),
    make_chunk(15, "the right of access to personal data undergoing processing", "1"),
    make_chunk(33, "notification of a personal data breach to the supervisory authority"),
    make_chunk(44, "general principle for transfers of personal data to third countries"),
]


@pytest.fixture
def populated_store():
    embedder = HashingEmbedder(dimensions=1024)
    store = ChunkStore()
    store.add(CORPUS, embedder.encode([c.text for c in CORPUS]))
    yield store, embedder
    store.close()


class TestPersistence:
    def test_length_reflects_stored_chunks(self, populated_store):
        store, _ = populated_store
        assert len(store) == len(CORPUS)

    def test_chunks_round_trip_with_their_metadata(self, populated_store):
        store, embedder = populated_store
        result = store.search(embedder.encode(["erasure without undue delay"]), k=1)[0]
        assert result.chunk.article == 17
        assert result.chunk.paragraph == "1"
        assert result.chunk.citation == "Article 17(1)"

    def test_survives_reopening_from_disk(self, tmp_path):
        embedder = HashingEmbedder(dimensions=256)
        path = tmp_path / "index.sqlite"
        with ChunkStore(path) as store:
            store.add(CORPUS, embedder.encode([c.text for c in CORPUS]))
            store.set_meta("embedder", embedder.name)
        with ChunkStore(path) as reopened:
            assert len(reopened) == len(CORPUS)
            assert reopened.get_meta("embedder") == "hashing-256"

    def test_adding_twice_accumulates(self, populated_store):
        store, embedder = populated_store
        store.add(CORPUS[:1], embedder.encode([CORPUS[0].text]))
        assert len(store) == len(CORPUS) + 1


class TestSearch:
    def test_returns_the_most_similar_chunk_first(self, populated_store):
        store, embedder = populated_store
        results = store.search(embedder.encode(["breach notification supervisory authority"]), k=2)
        assert results[0].chunk.article == 33

    def test_respects_k(self, populated_store):
        store, embedder = populated_store
        assert len(store.search(embedder.encode(["personal data"]), k=2)) == 2

    def test_k_larger_than_corpus_returns_everything(self, populated_store):
        store, embedder = populated_store
        assert len(store.search(embedder.encode(["personal data"]), k=99)) == len(CORPUS)

    def test_results_are_ordered_by_descending_score(self, populated_store):
        store, embedder = populated_store
        scores = [r.score for r in store.search(embedder.encode(["personal data"]), k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_no_results(self):
        with ChunkStore() as store:
            assert store.search(np.ones(8, dtype=np.float32), k=3) == []

    def test_newly_added_chunks_are_searchable(self, populated_store):
        # A stale cached matrix would hide the new chunk entirely.
        store, embedder = populated_store
        added = make_chunk(99, "records of processing activities maintained by the controller")
        store.add([added], embedder.encode([added.text]))
        results = store.search(embedder.encode(["records of processing activities"]), k=1)
        assert results[0].chunk.article == 99


class TestErrors:
    def test_mismatched_chunk_and_embedding_counts_are_rejected(self):
        with ChunkStore() as store, pytest.raises(ValueError, match="they must match"):
            store.add(CORPUS, np.zeros((2, 8), dtype=np.float32))

    def test_wrong_query_dimensions_are_rejected(self, populated_store):
        store, _ = populated_store
        with pytest.raises(ValueError, match="same embedder"):
            store.search(np.ones(7, dtype=np.float32), k=1)

    def test_k_below_one_is_rejected(self, populated_store):
        store, embedder = populated_store
        with pytest.raises(ValueError, match="k must be at least 1"):
            store.search(embedder.encode(["anything"]), k=0)

    def test_adding_nothing_is_a_no_op(self):
        with ChunkStore() as store:
            store.add([], np.zeros((0, 8), dtype=np.float32))
            assert len(store) == 0


class TestThreadSafety:
    """A store is built in one thread and queried from another whenever it is
    served — Streamlit, FastAPI, anything. sqlite3 forbids that by default,
    which surfaced as a hard crash the first time the demo was used.
    """

    def test_search_works_from_a_different_thread_than_the_builder(self):
        import threading

        built: list[ChunkStore] = []
        embedder = HashingEmbedder(dimensions=256)

        def build():
            store = ChunkStore()
            store.add(CORPUS, embedder.encode([c.text for c in CORPUS]))
            built.append(store)

        builder = threading.Thread(target=build)
        builder.start()
        builder.join()

        store = built[0]
        # This is the call that used to raise ProgrammingError.
        results = store.search(embedder.encode(["breach notification"]), k=1)
        assert results[0].chunk.article == 33
        store.close()

    def test_concurrent_searches_do_not_corrupt_results(self):
        import threading

        embedder = HashingEmbedder(dimensions=512)
        with ChunkStore() as store:
            store.add(CORPUS, embedder.encode([c.text for c in CORPUS]))
            found: list[int] = []

            def query():
                for _ in range(20):
                    hit = store.search(embedder.encode(["transfers to third countries"]), k=1)
                    found.append(hit[0].chunk.article)

            threads = [threading.Thread(target=query) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert len(found) == 80
            assert set(found) == {44}

    def test_writes_from_multiple_threads_all_land(self):
        import threading

        embedder = HashingEmbedder(dimensions=128)
        with ChunkStore() as store:

            def writer(index: int):
                chunk = make_chunk(index + 1, f"article text number {index}")
                store.add([chunk], embedder.encode([chunk.text]))

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert len(store) == 10
