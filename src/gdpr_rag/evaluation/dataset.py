"""Loading and validating the labelled evaluation set.

The labels are the ground truth every number in the README rests on, so this
module is strict about them: a malformed entry is an error, not a silently
skipped row. Quietly dropping a question would inflate every metric computed
afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[3] / "evaluation" / "questions.yaml"

_ARTICLE_LABEL = re.compile(r"^Article \d+$")


class Question(BaseModel):
    """One labelled question."""

    id: str
    question: str
    articles: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    unanswerable: bool = False
    note: str | None = None

    @field_validator("articles")
    @classmethod
    def _labels_are_article_level(cls, articles: list[str]) -> list[str]:
        for article in articles:
            if not _ARTICLE_LABEL.match(article):
                raise ValueError(
                    f"label {article!r} must be article-level, e.g. 'Article 17'. "
                    "Paragraph-level labels would make retrieval look wrong when it "
                    "returns the right article at a different paragraph."
                )
        return articles

    @field_validator("difficulty")
    @classmethod
    def _difficulty_is_known(cls, difficulty: str) -> str:
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        return difficulty

    @model_validator(mode="after")
    def _labels_match_answerability(self) -> Question:
        if self.unanswerable and self.articles:
            raise ValueError(f"{self.id}: an unanswerable question cannot have article labels")
        if not self.unanswerable and not self.articles:
            raise ValueError(
                f"{self.id}: an answerable question needs at least one article label "
                "(mark it unanswerable if the GDPR genuinely does not address it)"
            )
        return self

    @property
    def article_numbers(self) -> set[int]:
        """Label article numbers, for comparison against retrieved chunks."""
        return {int(a.split()[1]) for a in self.articles}


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[Question]:
    """Load the evaluation set, failing loudly on any malformed entry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no evaluation set at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    questions = [Question(**entry) for entry in raw]

    duplicates = {q.id for q in questions if sum(o.id == q.id for o in questions) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
    return questions
