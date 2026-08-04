"""Tests for the real-case evaluation set."""

import pytest

from gdpr_rag.evaluation import Case, evaluate_cases, load_cases


@pytest.fixture(scope="module")
def cases():
    return load_cases()


class TestShippedManifest:
    def test_loads(self, cases):
        assert len(cases) >= 60

    def test_most_carry_labels_from_the_court(self, cases):
        headnote = sum(c.label_source == "headnote" for c in cases)
        assert headnote / len(cases) > 0.7

    def test_some_cases_are_not_evaluable(self, cases):
        # Procedural judgments carry no substantive articles; excluding them
        # silently would misreport the sample size.
        assert any(not c.is_evaluable for c in cases)

    def test_labels_are_plausible_articles(self, cases):
        assert all(1 <= n <= 99 for c in cases for n in c.article_numbers)

    def test_manifest_records_provenance(self, cases):
        assert all(c.celex and c.source for c in cases)


class TestRedaction:
    def test_queries_drop_their_citations(self, cases):
        evaluable = [c for c in cases if c.is_evaluable]
        assert all("Article" not in c.query for c in evaluable)

    def test_raw_queries_keep_them(self, cases):
        leaky = [c for c in cases if c.is_evaluable and c.leak > 0]
        assert leaky, "expected some questions to name their own answer"
        assert any("Article" in c.raw_query for c in leaky)

    def test_the_set_leaks_substantially_when_unredacted(self, cases):
        evaluable = [c for c in cases if c.is_evaluable]
        leaking = sum(c.leak > 0 for c in evaluable)
        assert leaking / len(evaluable) > 0.4


class TestScoring:
    def make(self, cid, articles, question):
        return Case(id=cid, celex="x", articles=articles, questions=[question])

    def test_perfect_retrieval_scores_one(self):
        cases = [self.make("a", ["Article 17", "Article 5"], "erasure?")]
        report = evaluate_cases(cases, lambda q, k: [17, 5], k=5)
        assert report.coverage == 1.0
        assert report.found_all == 1.0

    def test_partial_retrieval_is_not_a_success(self):
        cases = [self.make("a", ["Article 17", "Article 5"], "erasure?")]
        report = evaluate_cases(cases, lambda q, k: [17, 99], k=5)
        assert report.coverage == 0.5
        assert report.found_any == 1.0
        assert report.found_all == 0.0

    def test_unevaluable_cases_are_excluded_and_counted(self):
        cases = [
            self.make("a", ["Article 17"], "erasure?"),
            Case(id="b", celex="y", articles=[], questions=["procedural?"]),
        ]
        report = evaluate_cases(cases, lambda q, k: [17], k=5)
        assert len(report.coverages) == 1
        assert report.excluded == 1

    def test_redaction_is_reported_in_the_summary(self):
        cases = [self.make("a", ["Article 17"], "under Article 17, erasure?")]
        assert "redacted" in str(evaluate_cases(cases, lambda q, k: [17], k=5))
        assert "raw" in str(evaluate_cases(cases, lambda q, k: [17], k=5, redacted=False))

    def test_worst_cases_are_surfaced_for_inspection(self):
        cases = [
            self.make("good", ["Article 17"], "erasure?"),
            self.make("bad", ["Article 99"], "something else?"),
        ]
        report = evaluate_cases(cases, lambda q, k: [17], k=5)
        assert report.worst(1)[0][0] == "bad"

    def test_scores_split_by_how_the_label_was_derived(self):
        cases = [
            Case(
                id="h",
                celex="x",
                articles=["Article 17"],
                questions=["q?"],
                label_source="headnote",
            ),
            Case(
                id="f",
                celex="y",
                articles=["Article 99"],
                questions=["q?"],
                label_source="frequency",
            ),
        ]
        report = evaluate_cases(cases, lambda q, k: [17], k=5)
        assert report.by_label_source["headnote"] == 1.0
        assert report.by_label_source["frequency"] == 0.0
