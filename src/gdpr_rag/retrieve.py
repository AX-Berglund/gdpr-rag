"""Query-time retrieval: embed a question, return the chunks nearest to it."""

from __future__ import annotations

from dataclasses import dataclass

from gdpr_rag.embed.base import Embedder
from gdpr_rag.store.sqlite_store import ChunkStore, SearchResult
from gdpr_rag.trace import Trace


@dataclass(frozen=True)
class Retriever:
    """Pairs an embedder with the store it built.

    Keeping them together is not incidental: an index is only queryable by the
    embedder that produced it, and separating them invites the mismatch.
    """

    embedder: Embedder
    store: ChunkStore

    def retrieve(self, question: str, k: int = 5, trace: Trace | None = None) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``question``.

        Passing a ``trace`` records the query, what came back and how long it
        took. Omitting it changes nothing.
        """
        if not question.strip():
            raise ValueError("question must not be empty")

        if trace is None:
            return self.store.search(self.embedder.encode([question]), k=k)

        with trace.span("retrieve", question=question, k=k, embedder=self.embedder.name) as span:
            results = self.store.search(self.embedder.encode([question]), k=k)
            span.record(
                citations=[r.chunk.citation for r in results],
                scores=[round(r.score, 3) for r in results],
            )
            return results

    @classmethod
    def build(
        cls,
        embedder: Embedder,
        chunks: list,
        path: str = ":memory:",
    ) -> Retriever:
        """Embed ``chunks`` and return a retriever over them.

        The embedder's name is recorded in the store so an index on disk can
        always report what built it.
        """
        store = ChunkStore(path)
        if chunks:
            store.add(chunks, embedder.encode([c.text for c in chunks]))
        store.set_meta("embedder", embedder.name)
        return cls(embedder=embedder, store=store)
