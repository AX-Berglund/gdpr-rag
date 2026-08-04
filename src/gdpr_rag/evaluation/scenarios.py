"""The narrative evaluation set.

Questions ask about a rule. Scenarios describe a situation and ask whether it
was allowed, which is what users actually send. The difference matters for
scoring: a question has one right article and several plausible neighbours, but
a scenario has a *set* of articles that must all be found. An answer that cites
the right and misses the exception is wrong in the way that matters most, so
these are scored on coverage rather than on whether anything correct appeared.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_SCENARIOS = Path(__file__).resolve().parents[3] / "evaluation" / "scenarios.yaml"

_ARTICLE_LABEL = re.compile(r"^Article \d+$")


class Scenario(BaseModel):
    """A situation, the articles needed to resolve it, and what is unstated."""

    id: str
    scenario: str
    articles: list[str]
    perspective: str = "neutral"
    difficulty: str = "medium"
    sub_questions: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    note: str | None = None

    @field_validator("articles")
    @classmethod
    def _labels_are_article_level(cls, articles: list[str]) -> list[str]:
        if not articles:
            raise ValueError("a scenario needs at least one article")
        for article in articles:
            if not _ARTICLE_LABEL.match(article):
                raise ValueError(f"label {article!r} must be article-level, e.g. 'Article 17'")
        return articles

    @field_validator("perspective")
    @classmethod
    def _perspective_is_known(cls, perspective: str) -> str:
        if perspective not in {"subject", "organisation", "neutral"}:
            raise ValueError(f"unknown perspective {perspective!r}")
        return perspective

    @property
    def article_numbers(self) -> set[int]:
        return {int(a.split()[1]) for a in self.articles}


def coverage(retrieved: list[int], required: set[int], k: int) -> float:
    """Fraction of the required articles present in the top ``k``.

    Hit rate would score a scenario as a success for finding one of four needed
    articles, which is precisely the failure this set exists to catch.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not required:
        raise ValueError("coverage is undefined without required articles")

    seen: dict[int, None] = {}
    for article in retrieved:
        seen.setdefault(article, None)
    return len(required & set(list(seen)[:k])) / len(required)


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS) -> list[Scenario]:
    """Load the narrative set, failing loudly on malformed entries."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no scenario set at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    scenarios = [Scenario(**entry) for entry in raw]

    duplicates = {s.id for s in scenarios if sum(o.id == s.id for o in scenarios) > 1}
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {sorted(duplicates)}")
    return scenarios
