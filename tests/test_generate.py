"""Tests for cited answer generation and the grounding check."""

import pytest

from gdpr_rag.generate import (
    Answer,
    answer_question,
    extract_citations,
    format_context,
)
from gdpr_rag.ingest.models import Chunk, ChunkKind
from gdpr_rag.store.sqlite_store import SearchResult


class ScriptedModel:
    """A language model that returns a fixed string, so tests assert on our logic."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def result(article: int, text: str, paragraph=None, point=None, score: float = 0.9):
    return SearchResult(
        chunk=Chunk(
            article=article,
            kind=ChunkKind.PARAGRAPH if paragraph else ChunkKind.BODY,
            text=text,
            title="Right to erasure" if article == 17 else None,
            paragraph=paragraph,
            point=point,
        ),
        score=score,
    )


RESULTS = [
    result(17, "the right to obtain erasure without undue delay", paragraph="1"),
    result(17, "the personal data are no longer necessary", paragraph="1", point="a"),
]


class TestExtractCitations:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("See Article 17.", ["Article 17"]),
            ("See Article 17(1).", ["Article 17(1)"]),
            ("See Article 17(1)(a).", ["Article 17(1)(a)"]),
            ("Both Article 4(7) and Article 15 apply.", ["Article 4(7)", "Article 15"]),
            ("No citation here.", []),
        ],
    )
    def test_citation_forms(self, text, expected):
        assert extract_citations(text) == expected

    def test_duplicates_collapse_but_order_is_kept(self):
        text = "Article 17(1) says X. Article 4 defines Y. Article 17(1) again."
        assert extract_citations(text) == ["Article 17(1)", "Article 4"]


class TestGrounding:
    def test_citation_matching_retrieval_is_supported(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Yes, see Article 17(1)."))
        assert answer.citations == ["Article 17(1)"]
        assert answer.unsupported_citations == []
        assert answer.is_grounded

    def test_invented_citation_is_flagged_not_silently_kept(self):
        answer = answer_question("q", RESULTS, ScriptedModel("See Article 99(2)."))
        assert answer.unsupported_citations == ["Article 99(2)"]
        assert answer.citations == []
        assert not answer.is_grounded

    def test_broader_citation_is_supported_by_a_narrower_retrieval(self):
        # Article 17(1)(a) was retrieved, so citing Article 17 is accurate.
        answer = answer_question("q", RESULTS, ScriptedModel("See Article 17."))
        assert answer.citations == ["Article 17"]

    def test_narrower_citation_is_not_supported_by_a_broader_retrieval(self):
        # Being shown Article 15 does not license a claim about Article 15(3).
        answer = answer_question(
            "q", [result(15, "right of access")], ScriptedModel("Article 15(3)")
        )
        assert answer.unsupported_citations == ["Article 15(3)"]

    def test_mixed_answer_separates_supported_from_invented(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Article 17(1) and Article 88."))
        assert answer.citations == ["Article 17(1)"]
        assert answer.unsupported_citations == ["Article 88"]
        assert not answer.is_grounded

    def test_answer_without_citations_is_not_grounded(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Data can sometimes be deleted."))
        assert not answer.is_grounded


class TestNoRetrieval:
    def test_empty_retrieval_refuses_without_calling_the_model(self):
        model = ScriptedModel("this should never be produced")
        answer = answer_question("q", [], model)
        assert model.last_prompt is None
        assert "cannot be answered" in answer.text
        assert not answer.is_grounded


class TestPrompt:
    def test_context_is_numbered_and_carries_citations(self):
        context = format_context(RESULTS)
        assert "[1] Article 17(1)" in context
        assert "[2] Article 17(1)(a)" in context

    def test_title_is_included_when_present(self):
        assert "Right to erasure" in format_context(RESULTS)

    def test_prompt_contains_context_and_question(self):
        model = ScriptedModel("Article 17(1)")
        answer_question("Can data be erased?", RESULTS, model)
        assert "Can data be erased?" in model.last_prompt
        assert "no longer necessary" in model.last_prompt

    def test_empty_question_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            answer_question("   ", RESULTS, ScriptedModel("x"))


class TestAnswerModel:
    def test_retrieved_records_everything_shown_to_the_model(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Article 17(1)"))
        assert answer.retrieved == ["Article 17(1)", "Article 17(1)(a)"]

    def test_defaults_are_empty(self):
        assert Answer(question="q", text="t").citations == []
