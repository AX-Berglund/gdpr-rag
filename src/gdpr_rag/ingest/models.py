"""Data model for a retrievable piece of the regulation.

A chunk carries the structural address it came from (article, paragraph,
point, definition) rather than only its text. Retrieval returns chunks, and an
answer cites their addresses, so the address has to survive the whole pipeline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    """The structural shape of the article a chunk came from.

    GDPR articles are not uniform. Most are numbered paragraphs, one (Article 4)
    is a list of definitions, and a handful are a single block of prose. Each
    needs different parsing, so the kind is recorded rather than inferred later.
    """

    PARAGRAPH = "paragraph"
    DEFINITION = "definition"
    BODY = "body"


class Chunk(BaseModel):
    """One retrievable unit of the regulation."""

    article: int = Field(description="Article number, e.g. 17 for the right to erasure")
    kind: ChunkKind
    text: str
    title: str | None = Field(default=None, description="Article title, when the source states one")
    paragraph: str | None = Field(default=None, description="Numbered paragraph, e.g. '2'")
    point: str | None = Field(
        default=None, description="Lettered point within a paragraph, e.g. 'a'"
    )
    definition: str | None = Field(default=None, description="Definition number within Article 4")

    @property
    def citation(self) -> str:
        """Human-readable address used when citing this chunk in an answer.

        Examples: ``Article 17(1)(a)``, ``Article 4(7)``, ``Article 1``.
        """
        if self.kind is ChunkKind.DEFINITION and self.definition is not None:
            return f"Article {self.article}({self.definition})"

        cite = f"Article {self.article}"
        if self.paragraph:
            cite += f"({self.paragraph})"
        if self.point:
            cite += f"({self.point})"
        return cite

    def __str__(self) -> str:
        return f"{self.citation}: {self.text[:60]}..."
