"""Retrieving for several phrasings of one question and merging the results.

A rewriter replaces the query with a better one. This does something different:
it issues *more than one* query and fuses what comes back.

The motivation is a measured failure. Asked whether contact details collected
at one event may be used to invite people to the next, retrieval returned
nothing that governed the question. Restating it in the regulation's own
register found the compatibility test; breaking it into its constituent legal
questions found the right to object to direct marketing. Neither found both,
because they fail in different directions — one shifts register, the other
shifts abstraction — and a single embedding of a multi-part question is a
blurred average of its parts.

Merging is by reciprocal rank fusion rather than by score. Similarity scores
from different queries are not on a common scale, so comparing them directly
would let whichever query happened to produce higher numbers dominate the
merge. Rank position is comparable across queries by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from gdpr_rag.generate import LanguageModel
from gdpr_rag.retrieve import Retriever
from gdpr_rag.store.sqlite_store import SearchResult
from gdpr_rag.trace import Trace

DECOMPOSE_PROMPT = """Break the question below into the separate legal questions that \
must each be answered to answer it fully. Do not answer them. Do not mention article \
numbers. Write each as a standalone question, one per line, at most {limit}.

Question: {question}

Sub-questions:"""

# The constant in reciprocal rank fusion, from the paper that introduced it. It
# damps the influence of the very top ranks so that a single query cannot carry
# the merge on its first result alone.
RRF_K = 60


def _clean(line: str) -> str:
    """Strip list decoration a model adds unbidden: bullets, "1.", quotes."""
    stripped = line.strip().lstrip("-•*").strip()
    number, separator, rest = stripped.partition(".")
    if separator and number.strip().isdigit():
        stripped = rest.strip()
    return stripped.strip('"').strip()


class Decomposer:
    """Splits a question into the sub-questions it is really asking.

    A scenario tends to contain several legal questions at once — is this
    processing lawful, was the person told, can they object. Retrieving for
    each separately keeps them from averaging into a single vague query.
    """

    def __init__(self, model: LanguageModel, limit: int = 3) -> None:
        self._model = model
        self._limit = limit

    @property
    def name(self) -> str:
        return f"decompose:{self._model.name}"

    def decompose(self, question: str) -> list[str]:
        """Return the sub-questions, always including the original.

        The original is kept because decomposition can lose the thread: a set
        of sub-questions may each be answerable while none of them is the
        question that was asked.
        """
        completion = self._model.complete(
            DECOMPOSE_PROMPT.format(question=question, limit=self._limit)
        )
        parts = [_clean(line) for line in completion.splitlines()]
        subs = [p for p in parts if p and p != question][: self._limit]
        return [question, *subs]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], k: int = RRF_K
) -> list[SearchResult]:
    """Merge ranked lists by summed reciprocal rank, best first.

    A chunk appearing modestly high in several rankings outranks one appearing
    top in a single ranking, which is the property we want: agreement across
    phrasings is evidence, and one query's enthusiasm is not.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            key = result.chunk.citation
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the highest-scoring instance so the reported similarity is
            # the best evidence found for that chunk, not the last seen.
            if key not in best or result.score > best[key].score:
                best[key] = result

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [best[key] for key in ordered]


def multi_query_retrieve(
    retriever: Retriever,
    queries: Sequence[str],
    k: int = 5,
    trace: Trace | None = None,
) -> list[SearchResult]:
    """Retrieve for every query and fuse the rankings.

    Each query retrieves ``k``, and the fused list is truncated to ``k``, so
    the caller is handed the same number of chunks as a single query would
    return. Costing more retrievals to surface a better ``k`` is the point;
    quietly returning more evidence than asked for would flatter the method in
    any measurement that scores a fixed budget.
    """
    if not queries:
        raise ValueError("multi-query retrieval needs at least one query")

    rankings = [retriever.retrieve(query, k=k, trace=trace) for query in queries]
    fused = reciprocal_rank_fusion(rankings)[:k]

    if trace is not None:
        with trace.span("fuse", queries=list(queries), k=k) as span:
            span.record(citations=[r.chunk.citation for r in fused])
    return fused
