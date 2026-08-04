"""Retrieval metrics.

Retrieval returns chunks, but questions are labelled with articles, so every
metric here works on the article sequence obtained by collapsing retrieved
chunks to their article number and keeping first appearance. Two chunks from
Article 17 are one hit at the rank of the earlier one, not two hits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def to_article_ranking(articles: Iterable[int]) -> list[int]:
    """Collapse a chunk-level ranking to a first-appearance article ranking."""
    seen: dict[int, None] = {}
    for article in articles:
        seen.setdefault(article, None)
    return list(seen)


def hit_rate(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if any labelled article appears in the top ``k``, else 0.0.

    Averaged over questions this is hit-rate@k: the share of questions where a
    correct article was retrieved at all. It is the metric that matters most
    here, because a generator cannot cite what retrieval never returned.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("hit rate is undefined for a question with no labelled articles")
    return float(bool(relevant & set(to_article_ranking(retrieved)[:k])))


def reciprocal_rank(retrieved: Sequence[int], relevant: set[int]) -> float:
    """``1 / rank`` of the first labelled article, or 0.0 if none appear.

    Averaged over questions this is MRR. Unlike hit rate it rewards putting the
    right article first rather than merely somewhere in the list.
    """
    if not relevant:
        raise ValueError("reciprocal rank is undefined for a question with no labelled articles")
    for rank, article in enumerate(to_article_ranking(retrieved), start=1):
        if article in relevant:
            return 1.0 / rank
    return 0.0
