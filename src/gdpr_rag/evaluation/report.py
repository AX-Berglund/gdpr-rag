"""Aggregating per-question scores into a reportable result.

Two decisions here are about honesty rather than arithmetic.

Unanswerable questions are excluded from retrieval means -- they have no
correct article, so scoring retrieval on them is meaningless -- but the count
of exclusions travels with the report, because a mean over an unstated subset
is how misleading numbers get published.

The per-difficulty breakdown is not decoration either. A headline hit rate can
look healthy while every hard question fails, and that is precisely the case
worth knowing about.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean

from gdpr_rag.evaluation.dataset import Question
from gdpr_rag.evaluation.metrics import hit_rate, ndcg, reciprocal_rank

Retrieve = Callable[[str, int], Sequence[int]]


@dataclass(frozen=True)
class QuestionResult:
    """Scores for one question."""

    question_id: str
    difficulty: str
    hit: float
    reciprocal_rank: float
    ndcg: float
    retrieved: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval quality for one configuration."""

    embedder: str
    k: int
    results: list[QuestionResult]
    excluded_unanswerable: int = 0
    phrasing: str = "formal"

    @property
    def hit_rate(self) -> float:
        return mean(r.hit for r in self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return mean(r.reciprocal_rank for r in self.results) if self.results else 0.0

    @property
    def ndcg(self) -> float:
        return mean(r.ndcg for r in self.results) if self.results else 0.0

    def by_difficulty(self) -> dict[str, float]:
        """Hit rate split by difficulty, hardest failures made visible."""
        return {
            level: mean(r.hit for r in self.results if r.difficulty == level)
            for level in ("easy", "medium", "hard")
            if any(r.difficulty == level for r in self.results)
        }

    def failures(self) -> list[QuestionResult]:
        """Questions where no correct article was retrieved at all."""
        return [r for r in self.results if r.hit == 0.0]

    def as_row(self) -> dict[str, object]:
        """One row of the results table published in the README."""
        return {
            "embedder": self.embedder,
            "k": self.k,
            "hit_rate": round(self.hit_rate, 3),
            "mrr": round(self.mrr, 3),
            "ndcg": round(self.ndcg, 3),
            "n": len(self.results),
            "phrasing": self.phrasing,
        }

    def __str__(self) -> str:
        by = ", ".join(f"{k} {v:.2f}" for k, v in self.by_difficulty().items())
        return (
            f"{self.embedder} @k={self.k}: "
            f"hit {self.hit_rate:.3f} · MRR {self.mrr:.3f} · nDCG {self.ndcg:.3f} "
            f"(n={len(self.results)}; {by})"
        )


def evaluate_retrieval(
    questions: Sequence[Question],
    retrieve: Retrieve,
    k: int = 5,
    embedder: str = "unknown",
    phrasing: str = "formal",
) -> RetrievalReport:
    """Score ``retrieve`` over the answerable questions in ``questions``.

    ``retrieve`` takes a question and ``k`` and returns the article numbers it
    found, in rank order. ``phrasing`` selects the well-formed question or the
    colloquial variant a real user would type.
    """
    answerable = [q for q in questions if not q.unanswerable]
    results = []
    for question in answerable:
        retrieved = list(retrieve(question.phrased(phrasing), k))
        relevant = question.article_numbers
        results.append(
            QuestionResult(
                question_id=question.id,
                difficulty=question.difficulty,
                hit=hit_rate(retrieved, relevant, k),
                reciprocal_rank=reciprocal_rank(retrieved, relevant),
                ndcg=ndcg(retrieved, relevant, k),
                retrieved=retrieved,
            )
        )
    return RetrievalReport(
        embedder=embedder,
        k=k,
        results=results,
        excluded_unanswerable=len(questions) - len(answerable),
        phrasing=phrasing,
    )
