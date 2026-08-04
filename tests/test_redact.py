"""Tests for query redaction."""

import pytest

from gdpr_rag.cases.redact import leak_score, redact, redact_articles


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "under Article 82 of the GDPR",
            "under Article 17(1)(a) of the GDPR",
            "under Articles 12 to 22 of the GDPR",
            "under Article 5(1)(e) and 6(1)(f)",
            "under article 82",
        ],
    )
    def test_citations_are_removed(self, text):
        assert "Article" not in redact_articles(text)
        assert "article" not in redact_articles(text)

    def test_substance_survives(self):
        out = redact_articles("Does Article 82 require that the applicant suffered harm?")
        assert "require that the applicant suffered harm?" in out

    def test_recitals_are_redacted_too(self):
        assert "recital" not in redact("as recital 47 explains").lower().replace("[recital]", "")

    def test_text_without_citations_is_untouched(self):
        text = "May a controller refuse a request that is manifestly excessive?"
        assert redact(text) == text


class TestLeakScore:
    def test_a_query_naming_its_answer_scores_one(self):
        assert leak_score("under Article 82 of the GDPR", {82}) == 1.0

    def test_a_query_naming_nothing_scores_zero(self):
        assert leak_score("does compensation require harm?", {82}) == 0.0

    def test_partial_leak_is_measured(self):
        assert leak_score("Article 6 but not the other", {6, 9}) == 0.5

    def test_ranges_leak_every_endpoint_named(self):
        assert leak_score("Articles 12 to 22", {12, 22}) == 1.0

    def test_no_labels_means_no_leak(self):
        assert leak_score("Article 82", set()) == 0.0
