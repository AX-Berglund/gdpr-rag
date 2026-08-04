"""Tests for query rewriting."""

import pytest

from gdpr_rag.rewrite import (
    HyDERewriter,
    NullRewriter,
    PerspectiveRewriter,
    apply_rewrite,
)
from gdpr_rag.trace import Trace


class ScriptedModel:
    def __init__(self, response: str = "rewritten text") -> None:
        self._response = response
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class FailingModel:
    @property
    def name(self) -> str:
        return "failing"

    def complete(self, prompt: str) -> str:
        raise RuntimeError("model unavailable")


class TestNullRewriter:
    def test_returns_the_question_unchanged(self):
        assert NullRewriter().rewrite("what are my rights?") == "what are my rights?"

    def test_is_named_so_it_appears_in_results_tables(self):
        assert NullRewriter().name == "none"


class TestHyDE:
    def test_keeps_the_question_alongside_the_draft(self):
        # A bad draft should not be able to drag retrieval somewhere unrelated.
        out = HyDERewriter(ScriptedModel("The controller shall erase.")).rewrite("delete my data")
        assert "delete my data" in out
        assert "The controller shall erase." in out

    def test_prompt_asks_for_regulation_style_prose(self):
        model = ScriptedModel()
        HyDERewriter(model).rewrite("delete my data")
        assert "General Data Protection Regulation" in model.calls[0]
        assert "delete my data" in model.calls[0]

    def test_empty_draft_falls_back_to_the_question(self):
        assert HyDERewriter(ScriptedModel("   ")).rewrite("delete my data") == "delete my data"

    def test_name_records_the_model(self):
        assert HyDERewriter(ScriptedModel()).name == "hyde:scripted"


class TestPerspective:
    def test_returns_the_restatement(self):
        out = PerspectiveRewriter(ScriptedModel("obligations of the controller to erase")).rewrite(
            "can i get my data deleted"
        )
        assert out == "obligations of the controller to erase"

    def test_empty_restatement_falls_back(self):
        assert PerspectiveRewriter(ScriptedModel("")).rewrite("q") == "q"

    def test_prompt_explains_the_obligation_framing(self):
        model = ScriptedModel()
        PerspectiveRewriter(model).rewrite("q")
        assert "obligations on controllers" in model.calls[0]


class TestApplyRewrite:
    def test_works_without_a_trace(self):
        assert apply_rewrite(NullRewriter(), "q") == "q"

    def test_records_before_and_after(self):
        trace = Trace("q")
        apply_rewrite(PerspectiveRewriter(ScriptedModel("restated")), "original", trace=trace)
        span = trace.find("rewrite")[0]
        assert span.inputs["question"] == "original"
        assert span.outputs["rewritten"] == "restated"
        assert span.outputs["changed"] is True

    def test_records_when_nothing_changed(self):
        trace = Trace("q")
        apply_rewrite(NullRewriter(), "original", trace=trace)
        assert trace.find("rewrite")[0].outputs["changed"] is False

    def test_a_failing_rewriter_falls_back_to_the_question(self):
        # Degrading to un-rewritten retrieval beats failing the whole query.
        assert apply_rewrite(HyDERewriter(FailingModel()), "original") == "original"

    def test_the_failure_is_recorded_rather_than_swallowed(self):
        trace = Trace("q")
        apply_rewrite(HyDERewriter(FailingModel()), "original", trace=trace)
        span = trace.find("rewrite")[0]
        assert "RuntimeError" in span.outputs["failed"]
        assert span.outputs["fell_back_to_original"] is True

    def test_empty_question_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            apply_rewrite(NullRewriter(), "   ")
