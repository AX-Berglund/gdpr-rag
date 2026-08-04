"""Tests for query tracing."""

import json

import pytest

from gdpr_rag.trace import MAX_VALUE_LENGTH, Trace


class TestStructure:
    def test_spans_are_recorded_in_order(self):
        trace = Trace("q")
        with trace.span("first"):
            pass
        with trace.span("second"):
            pass
        assert [s.name for s in trace.spans] == ["first", "second"]

    def test_spans_nest_to_match_the_work(self):
        trace = Trace("q")
        # Written nested on purpose: the indentation is what the test asserts.
        with trace.span("decompose"):  # noqa: SIM117
            with trace.span("retrieve"):
                with trace.span("embed"):
                    pass
        assert trace.spans[0].name == "decompose"
        assert trace.spans[0].children[0].name == "retrieve"
        assert trace.spans[0].children[0].children[0].name == "embed"

    def test_siblings_stay_siblings(self):
        trace = Trace("q")
        with trace.span("decompose"):
            with trace.span("retrieve"):
                pass
            with trace.span("retrieve"):
                pass
        assert len(trace.spans[0].children) == 2

    def test_stack_unwinds_so_later_spans_are_top_level(self):
        trace = Trace("q")
        with trace.span("outer"):  # noqa: SIM117
            with trace.span("inner"):
                pass
        with trace.span("after"):
            pass
        assert [s.name for s in trace.spans] == ["outer", "after"]


class TestRecording:
    def test_inputs_and_outputs_are_captured(self):
        trace = Trace("q")
        with trace.span("retrieve", k=5) as span:
            span.record(articles=[17, 5])
        assert trace.spans[0].inputs == {"k": 5}
        assert trace.spans[0].outputs == {"articles": [17, 5]}

    def test_record_can_be_called_more_than_once(self):
        trace = Trace("q")
        with trace.span("s") as span:
            span.record(a=1)
            span.record(b=2)
        assert trace.spans[0].outputs == {"a": 1, "b": 2}

    def test_long_strings_are_truncated(self):
        trace = Trace("q")
        with trace.span("generate") as span:
            span.record(prompt="x" * 5000)
        recorded = trace.spans[0].outputs["prompt"]
        assert len(recorded) < 400
        assert "+4700 chars" in recorded

    def test_long_lists_are_truncated(self):
        trace = Trace("q")
        with trace.span("retrieve") as span:
            span.record(articles=list(range(100)))
        assert len(trace.spans[0].outputs["articles"]) == 21
        assert "+80 more" in trace.spans[0].outputs["articles"][-1]

    def test_short_values_survive_intact(self):
        trace = Trace("q")
        with trace.span("s") as span:
            span.record(text="short")
        assert trace.spans[0].outputs["text"] == "short"

    def test_nested_structures_are_summarised_too(self):
        trace = Trace("q")
        with trace.span("s") as span:
            span.record(payload={"prompt": "y" * (MAX_VALUE_LENGTH + 50)})
        assert "…" in trace.spans[0].outputs["payload"]["prompt"]


class TestTiming:
    def test_duration_is_recorded(self):
        trace = Trace("q")
        with trace.span("s"):
            sum(range(100_000))
        assert trace.spans[0].duration_ms > 0

    def test_total_is_the_sum_of_top_level_spans(self):
        trace = Trace("q")
        with trace.span("a"):
            pass
        with trace.span("b"):
            pass
        assert trace.duration_ms == pytest.approx(
            trace.spans[0].duration_ms + trace.spans[1].duration_ms
        )


class TestErrors:
    def test_a_failing_span_records_the_error_and_re_raises(self):
        trace = Trace("q")
        with pytest.raises(ValueError, match="boom"), trace.span("bad"):
            raise ValueError("boom")
        assert trace.spans[0].error == "ValueError: boom"

    def test_failures_are_findable(self):
        trace = Trace("q")
        with pytest.raises(RuntimeError), trace.span("outer"), trace.span("inner"):
            raise RuntimeError("nested failure")
        assert [s.name for s in trace.failed()] == ["outer", "inner"]

    def test_the_stack_recovers_after_a_failure(self):
        # Otherwise every later span nests under the dead one.
        trace = Trace("q")
        with pytest.raises(ValueError), trace.span("bad"):
            raise ValueError("boom")
        with trace.span("after"):
            pass
        assert [s.name for s in trace.spans] == ["bad", "after"]


class TestQuerying:
    def test_find_locates_spans_at_any_depth(self):
        trace = Trace("q")
        with trace.span("decompose"):
            with trace.span("retrieve"):
                pass
            with trace.span("retrieve"):
                pass
        assert len(trace.find("retrieve")) == 2

    def test_find_returns_empty_for_unknown_names(self):
        assert Trace("q").find("absent") == []


class TestSerialisation:
    def test_round_trips_through_json(self):
        trace = Trace("can I get my data deleted?")
        with trace.span("retrieve", k=5) as span:
            span.record(articles=[17])
        data = json.loads(trace.to_json())
        assert data["query"] == "can I get my data deleted?"
        assert data["spans"][0]["outputs"]["articles"] == [17]

    def test_empty_sections_are_omitted(self):
        trace = Trace("q")
        with trace.span("s"):
            pass
        assert "inputs" not in trace.spans[0].to_dict()
        assert "children" not in trace.spans[0].to_dict()

    def test_summary_renders_a_readable_tree(self):
        trace = Trace("q")
        with trace.span("decompose") as span:
            span.record(sub_questions=2)
            with trace.span("retrieve"):
                pass
        summary = trace.summary()
        assert "decompose" in summary
        assert "sub_questions: 2" in summary
        assert summary.index("decompose") < summary.index("retrieve")

    def test_summary_marks_failures(self):
        trace = Trace("q")
        with pytest.raises(ValueError), trace.span("bad"):
            raise ValueError("boom")
        assert "✗" in trace.summary()
