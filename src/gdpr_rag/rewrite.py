"""Turning what a user typed into something the corpus can match.

The regulation is written in one register and users write in another. Someone
types `how do i get a company to delete my data`; the text says *"The data
subject shall have the right to obtain from the controller the erasure of
personal data concerning him or her without undue delay"*. Measured, that gap
costs about ten points of hit@5, and it is far wider for questions asked from
the individual's side (0.333) than the organisation's (0.784) — because the
regulation states obligations on controllers, so an organisation's question is
already halfway to the corpus register.

Every rewriter here is swappable and every one is measured against
``NullRewriter``, which does nothing. A rewriting step that cannot beat leaving
the query alone is a cost with no benefit, and that has to be demonstrable
rather than assumed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gdpr_rag.generate import LanguageModel
from gdpr_rag.trace import Trace

HYDE_PROMPT = """Write a short passage from the EU General Data Protection Regulation \
that would answer the question below. Match the drafting style of the regulation: \
formal, obligation-oriented, referring to "the controller", "the processor" and "the \
data subject".

Do not answer the question, and do not mention article numbers. Write two or three \
sentences of regulation-style text only.

Question: {question}

Passage:"""

PERSPECTIVE_PROMPT = """Rewrite the question below in the language the EU General Data \
Protection Regulation itself uses.

The regulation states obligations on controllers, so a question asked by an individual \
about their own rights should be restated in terms of what a controller must do. Keep \
the meaning exactly; change only the vocabulary and framing. Use terms such as \
"personal data", "controller", "processor", "data subject", "processing".

Reply with the rewritten question and nothing else.

Question: {question}

Rewritten:"""


@runtime_checkable
class Rewriter(Protocol):
    """Transforms a user's question into the text actually embedded."""

    @property
    def name(self) -> str: ...

    def rewrite(self, question: str) -> str: ...


class NullRewriter:
    """Leaves the question alone. The baseline every other rewriter must beat."""

    @property
    def name(self) -> str:
        return "none"

    def rewrite(self, question: str) -> str:
        return question


class HyDERewriter:
    """Embeds a hypothetical regulation passage instead of the question.

    The insight is that a question and an answer are different kinds of text,
    so comparing a question's embedding against a corpus of statutory prose
    compares unlike with unlike. Asking a model to draft what the regulation
    *would* say produces text in the corpus's own register, and matching prose
    against prose is an easier problem.

    The draft is often wrong on substance. That does not matter — it is never
    shown to anyone, and is used only to find real text.
    """

    def __init__(self, model: LanguageModel) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"hyde:{self._model.name}"

    def rewrite(self, question: str) -> str:
        drafted = self._model.complete(HYDE_PROMPT.format(question=question)).strip()
        # Keeping the question alongside the draft hedges against a bad draft
        # sending retrieval somewhere unrelated.
        return f"{question}\n\n{drafted}" if drafted else question


class PerspectiveRewriter:
    """Restates a question in the regulation's obligation-side vocabulary.

    Cheaper than HyDE — it produces one sentence rather than a paragraph — and
    aimed squarely at the measured asymmetry between how individuals and
    organisations ask.
    """

    def __init__(self, model: LanguageModel) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"perspective:{self._model.name}"

    def rewrite(self, question: str) -> str:
        restated = self._model.complete(PERSPECTIVE_PROMPT.format(question=question)).strip()
        return restated or question


def apply_rewrite(
    rewriter: Rewriter,
    question: str,
    trace: Trace | None = None,
) -> str:
    """Rewrite ``question``, recording the before and after when tracing.

    A rewriter that fails must not take the query down with it: retrieval on
    the original question is worse than nothing happening at all, but it is far
    better than an error page. The failure is recorded rather than swallowed.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    if trace is None:
        try:
            return rewriter.rewrite(question)
        except Exception:
            return question

    with trace.span("rewrite", strategy=rewriter.name, question=question) as span:
        try:
            rewritten = rewriter.rewrite(question)
        except Exception as exc:
            span.record(failed=f"{type(exc).__name__}: {exc}", fell_back_to_original=True)
            return question
        span.record(rewritten=rewritten, changed=rewritten != question)
        return rewritten
