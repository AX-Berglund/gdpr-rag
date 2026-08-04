"""Answering questions from retrieved chunks, with citations that are checked.

The failure mode this module is built around is not the model writing prose
that reads badly -- it is the model citing an article it was never shown.
Asking for citations in the prompt makes them appear; it does not make them
true. So every citation the model emits is verified against what retrieval
actually returned, and the ones that are not are reported rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from gdpr_rag.store.sqlite_store import SearchResult
from gdpr_rag.trace import Trace

# Matches "Article 17", "Article 17(1)", "Article 17(1)(a)", "Article 4(7)".
_CITATION = re.compile(r"Article\s+\d+(?:\(\w+\))*")

PROMPT_TEMPLATE = """You are answering questions about the General Data Protection \
Regulation (GDPR). Use only the numbered excerpts below. If they do not contain the \
answer, say so plainly rather than drawing on outside knowledge.

Cite the article after each claim, in the form "Article 17(1)(a)". Cite only articles \
that appear in the excerpts.

Excerpts:
{context}

Question: {question}

Answer:"""


@runtime_checkable
class LanguageModel(Protocol):
    """Any text-in, text-out model. Kept minimal so backends stay swappable."""

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str) -> str: ...


class Answer(BaseModel):
    """A generated answer together with the evidence behind it."""

    question: str
    text: str
    citations: list[str] = Field(
        default_factory=list, description="Article citations that retrieval supports"
    )
    unsupported_citations: list[str] = Field(
        default_factory=list,
        description="Citations the model produced that retrieval did not return",
    )
    retrieved: list[str] = Field(
        default_factory=list, description="Citations of every chunk given to the model"
    )

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites something and invents nothing.

        An answer with no citations is not grounded either: on this corpus a
        claim without an article behind it is exactly what we are trying to
        avoid emitting.
        """
        return bool(self.citations) and not self.unsupported_citations


def format_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved chunks as numbered, citable excerpts."""
    lines = []
    for i, result in enumerate(results, start=1):
        title = f" — {result.chunk.title}" if result.chunk.title else ""
        lines.append(f"[{i}] {result.chunk.citation}{title}\n{result.chunk.text}")
    return "\n\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """Pull article citations out of generated text, preserving first-seen order."""
    seen: dict[str, None] = {}
    for match in _CITATION.findall(text):
        seen.setdefault(re.sub(r"\s+", " ", match).strip(), None)
    return list(seen)


def _supports(citation: str, retrieved: set[str]) -> bool:
    """Whether ``citation`` is backed by a retrieved chunk.

    A broader citation is supported by a narrower one: if Article 17(1)(a) was
    retrieved then citing Article 17 is accurate, merely less precise. The
    reverse is not true -- being shown Article 17 does not license a claim
    about Article 17(1)(a) specifically.
    """
    return any(r == citation or r.startswith(f"{citation}(") for r in retrieved)


def answer_question(
    question: str,
    results: Sequence[SearchResult],
    model: LanguageModel,
    trace: Trace | None = None,
) -> Answer:
    """Generate an answer from retrieved chunks and verify its citations.

    Passing a ``trace`` records the prompt, the raw completion and the
    grounding verdict — which is what makes a wrong answer diagnosable rather
    than merely wrong.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    retrieved = [r.chunk.citation for r in results]
    if not results:
        # Refusing here rather than prompting the model keeps a known-unanswerable
        # question from becoming a fluent guess.
        return Answer(
            question=question,
            text="No relevant articles were retrieved, so this cannot be answered from the corpus.",
            retrieved=[],
        )

    prompt = PROMPT_TEMPLATE.format(context=format_context(results), question=question)
    if trace is None:
        text = model.complete(prompt)
    else:
        with trace.span("generate", model=model.name, evidence=retrieved) as span:
            text = model.complete(prompt)
            # The prompt is the template (static, in source) wrapped around the
            # chunks already listed in `evidence`, so recording its text would
            # copy the corpus into every trace to say nothing new. Its size is
            # the part that varies and matters.
            span.record(prompt_chars=len(prompt), completion=text)

    retrieved_set = set(retrieved)
    cited = extract_citations(text)
    answer = Answer(
        question=question,
        text=text,
        citations=[c for c in cited if _supports(c, retrieved_set)],
        unsupported_citations=[c for c in cited if not _supports(c, retrieved_set)],
        retrieved=retrieved,
    )
    if trace is not None:
        with trace.span("verify_citations", cited=cited) as span:
            span.record(
                supported=answer.citations,
                unsupported=answer.unsupported_citations,
                grounded=answer.is_grounded,
            )
    return answer
