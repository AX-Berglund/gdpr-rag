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

from gdpr_rag.evaluation._paths import evaluation_file

DEFAULT_QUESTIONS = evaluation_file("questions.yaml")

_ARTICLE_LABEL = re.compile(r"^Article \d+$")


class Question(BaseModel):
    """One labelled question."""

    id: str
    question: str
    articles: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    unanswerable: bool = False
    note: str | None = None
    colloquial: str | None = Field(
        default=None,
        description="The same question phrased as a user would actually type it",
    )
    perspective: str = Field(
        default="neutral",
        description="Who is asking: subject, organisation, or neutral",
    )

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

    @field_validator("perspective")
    @classmethod
    def _perspective_is_known(cls, perspective: str) -> str:
        if perspective not in {"subject", "organisation", "neutral"}:
            raise ValueError(f"unknown perspective {perspective!r}")
        return perspective

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

    def phrased(self, style: str = "formal") -> str:
        """The question in the requested style.

        Falls back to the formal phrasing rather than raising, so a question
        without a colloquial variant is still scored instead of silently
        dropping out of the set and shifting the mean.
        """
        if style not in {"formal", "colloquial"}:
            raise ValueError(f"unknown phrasing style {style!r}")
        if style == "colloquial" and self.colloquial:
            return self.colloquial
        return self.question

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
