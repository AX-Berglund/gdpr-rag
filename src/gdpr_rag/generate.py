"""Answering questions from retrieved chunks, with citations that are checked.

The failure mode this module is built around is not the model writing prose
that reads badly -- it is the model citing an article it was never shown.
Asking for citations in the prompt makes them appear; it does not make them
true. So every citation the model emits is verified against what retrieval
actually returned, and the ones that are not are reported rather than hidden.

Verification alone turned out not to be enough. Shown Article 13(3) and asked
to cite it, a model paraphrased it correctly and wrote "Article 3(3)" -- one
digit dropped, and 3(3) is a real article about territorial scope, so the
result was a plausible citation attached to the wrong provision. It was caught,
but only after the fact.

So the model is asked to cite by excerpt number instead: it writes ``[2]`` and
the article citation is substituted in afterwards from what retrieval returned.
A number it was handed cannot be misspelt into a different article, which makes
the common failure impossible rather than merely detectable. Article numbers
written out anyway are still checked, because instructions are not guarantees.
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

# Matches the "[2]" markers the model is asked to cite with, absorbing a
# preceding "excerpt"/"excerpts" so that "see excerpts [3] and [4]" resolves to
# "see Article 13(3) and Article 14(4)" rather than leaving the word stranded
# in front of a citation that no longer needs it.
_EXCERPT = re.compile(r"(?:\bexcerpts?\s+)?(\[(\d+)\])", re.IGNORECASE)

PROMPT_TEMPLATE = """You are answering questions about the General Data Protection \
Regulation (GDPR). Use only the numbered excerpts below. If they do not contain the \
answer, say so plainly rather than drawing on outside knowledge.

Cite by excerpt number in square brackets, like [2], immediately after the claim it \
supports. Do not write article numbers yourself: the excerpt number is replaced with \
the correct article citation afterwards, so [2] cites excerpt 2 accurately and writing \
the number out risks getting it wrong.

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


def resolve_excerpts(
    text: str, results: Sequence[SearchResult]
) -> tuple[str, list[str], list[str]]:
    """Substitute ``[n]`` markers with the citation of the n-th excerpt.

    Returns the rewritten text, the citations resolved in first-seen order, and
    any markers pointing outside the evidence. An out-of-range marker is left
    in place rather than quietly dropped: the claim beside it rests on nothing,
    and the reader should see that.
    """
    citations = [r.chunk.citation for r in results]
    resolved: dict[str, None] = {}
    invalid: dict[str, None] = {}

    def substitute(match: re.Match[str]) -> str:
        index = int(match.group(2))
        if 1 <= index <= len(citations):
            citation = citations[index - 1]
            resolved.setdefault(citation, None)
            return citation
        # Out of range: leave the text exactly as written, and report the
        # marker itself rather than whatever prose happened to precede it.
        invalid.setdefault(match.group(1), None)
        return match.group(0)

    return _EXCERPT.sub(substitute, text), list(resolved), list(invalid)


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
    rendered, resolved, invalid = resolve_excerpts(text, results)

    # Article numbers the model wrote out despite being asked not to. These are
    # the ones that can be wrong, so they are still checked against retrieval —
    # read from the raw completion, since substitution has by now put correct
    # citations into the rendered text.
    written = extract_citations(text)
    supported = [c for c in written if _supports(c, retrieved_set)]

    citations = list(dict.fromkeys(resolved + supported))
    unsupported = [c for c in written if not _supports(c, retrieved_set)]
    # A marker pointing past the evidence cites something that does not exist.
    unsupported += [f"excerpt {marker}" for marker in invalid]

    answer = Answer(
        question=question,
        text=rendered,
        citations=citations,
        unsupported_citations=unsupported,
        retrieved=retrieved,
    )
    if trace is not None:
        with trace.span("verify_citations", cited=citations + unsupported) as span:
            span.record(
                supported=answer.citations,
                unsupported=answer.unsupported_citations,
                grounded=answer.is_grounded,
                resolved_from_excerpts=resolved,
            )
    return answer
