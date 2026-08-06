"""Tests for cited answer generation and the grounding check."""

import pytest

from gdpr_rag.generate import (
    Answer,
    answer_question,
    extract_citations,
    format_context,
    resolve_excerpts,
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


class TestResolveExcerpts:
    """Citing by excerpt number, so a citation cannot be mistyped into another.

    This exists because of a real answer: shown Article 13(3), the model
    paraphrased it correctly and cited "Article 3(3)" — one digit dropped, onto
    a real article about territorial scope. Verification caught it, but the
    answer was already wrong. A number the model was handed cannot drift.
    """

    def test_a_marker_becomes_the_citation_of_that_excerpt(self):
        text, resolved, invalid = resolve_excerpts("Erasure applies [1].", RESULTS)
        assert text == "Erasure applies Article 17(1)."
        assert resolved == ["Article 17(1)"]
        assert invalid == []

    def test_each_marker_resolves_independently(self):
        text, resolved, _ = resolve_excerpts("First [1], then [2].", RESULTS)
        assert text == "First Article 17(1), then Article 17(1)(a)."
        assert resolved == ["Article 17(1)", "Article 17(1)(a)"]

    def test_repeated_markers_collapse_but_every_mention_is_substituted(self):
        text, resolved, _ = resolve_excerpts("[1] and again [1].", RESULTS)
        assert text == "Article 17(1) and again Article 17(1)."
        assert resolved == ["Article 17(1)"]

    @pytest.mark.parametrize("marker", ["[0]", "[3]", "[99]"])
    def test_a_marker_outside_the_evidence_is_flagged_and_left_visible(self, marker):
        # Left in place deliberately: the claim beside it rests on nothing, and
        # silently deleting the marker would hide that from the reader.
        text, resolved, invalid = resolve_excerpts(f"Claimed {marker}.", RESULTS)
        assert text == f"Claimed {marker}."
        assert resolved == []
        assert invalid == [marker]

    @pytest.mark.parametrize(
        "written,expected",
        [
            ("see excerpt [1]", "see Article 17(1)"),
            ("see excerpts [1] and [2]", "see Article 17(1) and Article 17(1)(a)"),
            ("see Excerpt [1]", "see Article 17(1)"),
        ],
    )
    def test_the_word_excerpt_is_absorbed_rather_than_left_stranded(self, written, expected):
        # Models write "as set out in excerpts [3] and [4]"; substituting the
        # marker alone leaves "in excerpts Article 13(3) and Article 14(4)".
        assert resolve_excerpts(written, RESULTS)[0] == expected

    def test_an_absorbed_word_still_reports_only_the_marker(self):
        _, _, invalid = resolve_excerpts("see excerpt [9]", RESULTS)
        assert invalid == ["[9]"]

    def test_text_without_markers_is_untouched(self):
        text, resolved, invalid = resolve_excerpts("No markers at all.", RESULTS)
        assert (text, resolved, invalid) == ("No markers at all.", [], [])


class TestCitingByExcerpt:
    def test_a_cited_excerpt_grounds_the_answer(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Data may be erased [1]."))
        assert answer.text == "Data may be erased Article 17(1)."
        assert answer.citations == ["Article 17(1)"]
        assert answer.is_grounded

    def test_the_model_cannot_mistype_an_article_it_never_writes(self):
        # The digit-drop that motivated this: citing by number yields the right
        # article even though the model wrote nothing resembling one.
        answer = answer_question("q", RESULTS, ScriptedModel("Inform them first [2]."))
        assert "Article 17(1)(a)" in answer.text
        assert answer.unsupported_citations == []

    def test_a_marker_past_the_evidence_is_not_grounded(self):
        answer = answer_question("q", RESULTS, ScriptedModel("As shown [7]."))
        assert answer.unsupported_citations == ["excerpt [7]"]
        assert not answer.is_grounded

    def test_markers_and_written_articles_are_both_accounted_for(self):
        answer = answer_question("q", RESULTS, ScriptedModel("See [1], and Article 88."))
        assert answer.citations == ["Article 17(1)"]
        assert answer.unsupported_citations == ["Article 88"]
        assert not answer.is_grounded

    def test_a_marker_and_the_same_article_written_out_are_not_double_counted(self):
        answer = answer_question("q", RESULTS, ScriptedModel("See [1] i.e. Article 17(1)."))
        assert answer.citations == ["Article 17(1)"]


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


class TestTracing:
    def test_generation_without_a_trace_is_unchanged(self):
        answer = answer_question("q", RESULTS, ScriptedModel("Article 17(1)"))
        assert answer.citations == ["Article 17(1)"]

    def test_a_trace_records_the_prompt_and_completion(self):
        from gdpr_rag.trace import Trace

        trace = Trace("q")
        answer_question("q", RESULTS, ScriptedModel("See Article 17(1)."), trace=trace)
        span = trace.find("generate")[0]
        assert span.inputs["model"] == "scripted"
        assert "Article 17(1)" in span.outputs["completion"]
        # The evidence identifies what was shown; the prompt text would just
        # copy the corpus into the trace.
        assert span.inputs["evidence"] == ["Article 17(1)", "Article 17(1)(a)"]
        assert span.outputs["prompt_chars"] > 100

    def test_the_grounding_verdict_is_traced(self):
        from gdpr_rag.trace import Trace

        trace = Trace("q")
        answer_question("q", RESULTS, ScriptedModel("See Article 99(2)."), trace=trace)
        span = trace.find("verify_citations")[0]
        assert span.outputs["unsupported"] == ["Article 99(2)"]
        assert span.outputs["grounded"] is False

    def test_a_refusal_records_no_generate_span(self):
        from gdpr_rag.trace import Trace

        # The model is never called, so a generate span would be a lie.
        trace = Trace("q")
        answer_question("q", [], ScriptedModel("unused"), trace=trace)
        assert trace.find("generate") == []
