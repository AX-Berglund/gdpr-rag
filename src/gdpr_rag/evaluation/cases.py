"""The real-case evaluation set.

Court of Justice judgments, with labels the Court supplied rather than an
author. This is the only set here whose ground truth is external, which makes
it the most trustworthy and by some distance the hardest — these cases exist
because the law was unclear enough to reach the Court.

Queries are the referring court's own questions, **redacted**: those questions
routinely name the provision they ask about, and 52% of this set gives away at
least one of its own answers. Redaction turned out to matter little for
retrieval, because article numbers live in chunk metadata rather than embedded
text, but it matters a great deal for generation, which reads the citation.

Scored on coverage, like the narrative scenarios: a judgment turning on four
articles is not answered by finding one of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml
from pydantic import BaseModel, Field

from gdpr_rag.cases.redact import leak_score, redact
from gdpr_rag.evaluation.scenarios import coverage

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "evaluation" / "cases.yaml"


class Case(BaseModel):
    """One judgment, its labels, and the questions that were referred."""

    id: str
    celex: str
    court: str = "CJEU"
    title: str = ""
    articles: list[str] = Field(default_factory=list)
    label_source: str = "headnote"
    questions: list[str] = Field(default_factory=list)
    mentions: dict[int, int] = Field(default_factory=dict)
    source: str = ""

    @property
    def article_numbers(self) -> set[int]:
        return {int(a.split()[1]) for a in self.articles}

    @property
    def query(self) -> str:
        """The referred questions with citations removed."""
        return redact(" ".join(self.questions))

    @property
    def raw_query(self) -> str:
        """The referred questions as written, citations intact."""
        return " ".join(self.questions)

    @property
    def leak(self) -> float:
        """How much of the answer the unredacted question gives away."""
        return leak_score(self.raw_query, self.article_numbers)

    @property
    def is_evaluable(self) -> bool:
        """Whether this case can be scored at all.

        Some judgments concern only supervisory-authority competence, and carry
        no substantive articles; others are decided without a referred question
        this parser could recover. Both are kept in the manifest and excluded
        from scoring, with the exclusion counted.
        """
        return bool(self.articles and self.questions)


@dataclass(frozen=True)
class CaseReport:
    """Coverage over the real-case set."""

    k: int
    retriever: str
    coverages: list[tuple[str, float]]
    by_label_source: dict[str, float]
    excluded: int
    mean_leak: float
    redacted: bool = True

    @property
    def coverage(self) -> float:
        return mean(c for _, c in self.coverages) if self.coverages else 0.0

    @property
    def found_any(self) -> float:
        return mean(c > 0 for _, c in self.coverages) if self.coverages else 0.0

    @property
    def found_all(self) -> float:
        return mean(c == 1.0 for _, c in self.coverages) if self.coverages else 0.0

    def worst(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.coverages, key=lambda pair: pair[1])[:n]

    def __str__(self) -> str:
        state = "redacted" if self.redacted else "raw"
        return (
            f"{self.retriever} @k={self.k} ({state}): coverage {self.coverage:.3f} · "
            f"any {self.found_any:.3f} · all {self.found_all:.3f} "
            f"(n={len(self.coverages)}, excluded {self.excluded})"
        )


def load_cases(path: str | Path = DEFAULT_CASES) -> list[Case]:
    """Load the generated case manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no case manifest at {path}. Run: python scripts/fetch_cases.py --discover"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Case(**entry) for entry in raw]


def evaluate_cases(
    cases: Sequence[Case],
    retrieve: Callable[[str, int], Sequence[int]],
    k: int = 10,
    retriever: str = "unknown",
    redacted: bool = True,
) -> CaseReport:
    """Score retrieval over the real-case set.

    ``redacted`` selects whether queries keep their article citations. Running
    both ways is how the leak is measured rather than assumed.
    """
    evaluable = [c for c in cases if c.is_evaluable]
    scores, per_source = [], {}
    for case in evaluable:
        query = case.query if redacted else case.raw_query
        found = coverage(list(retrieve(query, k)), case.article_numbers, k)
        scores.append((case.id, found))
        per_source.setdefault(case.label_source, []).append(found)

    return CaseReport(
        k=k,
        retriever=retriever,
        coverages=scores,
        by_label_source={source: mean(v) for source, v in per_source.items()},
        excluded=len(cases) - len(evaluable),
        mean_leak=mean(c.leak for c in evaluable) if evaluable else 0.0,
        redacted=redacted,
    )
