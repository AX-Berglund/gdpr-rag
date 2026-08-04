"""Tests for evaluation aggregation."""

import pytest

from gdpr_rag.evaluation.dataset import Question
from gdpr_rag.evaluation.report import evaluate_retrieval

QUESTIONS = [
    Question(id="a", question="erasure", articles=["Article 17"], difficulty="easy"),
    Question(id="b", question="breach", articles=["Article 33"], difficulty="medium"),
    Question(id="c", question="transfers", articles=["Article 44"], difficulty="hard"),
    Question(id="d", question="tax rates", articles=[], unanswerable=True, difficulty="easy"),
]

PERFECT = {"erasure": [17], "breach": [33], "transfers": [44]}
HARD_FAILS = {"erasure": [17], "breach": [33], "transfers": [1, 2, 3]}


def fixed(table):
    return lambda question, k: table.get(question, [])[:k]


class TestAggregation:
    def test_perfect_retrieval_scores_one(self):
        report = evaluate_retrieval(QUESTIONS, fixed(PERFECT), k=3)
        assert report.hit_rate == 1.0
        assert report.mrr == 1.0
        assert report.ndcg == pytest.approx(1.0)

    def test_unanswerable_questions_are_excluded_and_counted(self):
        report = evaluate_retrieval(QUESTIONS, fixed(PERFECT), k=3)
        assert len(report.results) == 3
        assert report.excluded_unanswerable == 1

    def test_partial_failure_lowers_the_mean(self):
        report = evaluate_retrieval(QUESTIONS, fixed(HARD_FAILS), k=3)
        assert report.hit_rate == pytest.approx(2 / 3)


class TestVisibility:
    def test_difficulty_breakdown_exposes_where_it_fails(self):
        report = evaluate_retrieval(QUESTIONS, fixed(HARD_FAILS), k=3)
        breakdown = report.by_difficulty()
        assert breakdown["easy"] == 1.0
        assert breakdown["hard"] == 0.0

    def test_failures_are_listed_for_inspection(self):
        report = evaluate_retrieval(QUESTIONS, fixed(HARD_FAILS), k=3)
        assert [f.question_id for f in report.failures()] == ["c"]

    def test_row_carries_the_sample_size(self):
        row = evaluate_retrieval(QUESTIONS, fixed(PERFECT), k=3, embedder="hashing-512").as_row()
        assert row["n"] == 3
        assert row["embedder"] == "hashing-512"

    def test_summary_mentions_the_configuration(self):
        report = evaluate_retrieval(QUESTIONS, fixed(PERFECT), k=3, embedder="minilm")
        assert "minilm" in str(report)
        assert "k=3" in str(report)


class TestEdgeCases:
    def test_retrieving_nothing_scores_zero_without_crashing(self):
        report = evaluate_retrieval(QUESTIONS, fixed({}), k=3)
        assert report.hit_rate == 0.0
        assert report.mrr == 0.0

    def test_all_unanswerable_gives_an_empty_report(self):
        only_unanswerable = [q for q in QUESTIONS if q.unanswerable]
        report = evaluate_retrieval(only_unanswerable, fixed(PERFECT), k=3)
        assert report.results == []
        assert report.hit_rate == 0.0

    def test_k_limits_what_retrieval_may_return(self):
        # Retrieval hands back a long list; only the top k may count.
        report = evaluate_retrieval([QUESTIONS[0]], lambda q, k: [1, 2, 3, 17][:k], k=2)
        assert report.hit_rate == 0.0
