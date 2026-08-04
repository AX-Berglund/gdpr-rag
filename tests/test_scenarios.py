"""Tests for the narrative evaluation set and coverage scoring."""

import pytest
from pydantic import ValidationError

from gdpr_rag.evaluation import Scenario, coverage, load_scenarios


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


class TestShippedSet:
    def test_loads(self, scenarios):
        assert len(scenarios) >= 15

    def test_ids_are_unique(self, scenarios):
        assert len({s.id for s in scenarios}) == len(scenarios)

    def test_every_scenario_needs_several_articles(self, scenarios):
        # A single-article scenario is just a question with extra words.
        assert all(len(s.articles) >= 2 for s in scenarios)

    def test_every_scenario_decomposes(self, scenarios):
        assert all(len(s.sub_questions) >= 2 for s in scenarios)

    def test_missing_facts_are_recorded(self, scenarios):
        assert all(s.missing_facts for s in scenarios)

    def test_both_sides_are_represented(self, scenarios):
        perspectives = {s.perspective for s in scenarios}
        assert "subject" in perspectives and "organisation" in perspectives

    def test_labels_are_plausible_articles(self, scenarios):
        assert all(1 <= n <= 99 for s in scenarios for n in s.article_numbers)


class TestValidation:
    def test_a_scenario_without_articles_is_rejected(self):
        with pytest.raises(ValidationError, match="at least one article"):
            Scenario(id="x", scenario="story", articles=[])

    def test_paragraph_level_labels_are_rejected(self):
        with pytest.raises(ValidationError, match="article-level"):
            Scenario(id="x", scenario="story", articles=["Article 17(3)"])

    def test_unknown_perspective_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown perspective"):
            Scenario(id="x", scenario="s", articles=["Article 5"], perspective="judge")

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scenarios(tmp_path / "absent.yaml")


class TestCoverage:
    def test_all_required_articles_found(self):
        assert coverage([17, 5, 6], {17, 5, 6}, k=3) == 1.0

    def test_partial_scores_the_fraction(self):
        assert coverage([17, 1, 2], {17, 5, 6}, k=3) == pytest.approx(1 / 3)

    def test_nothing_found_scores_zero(self):
        assert coverage([1, 2, 3], {17, 5}, k=3) == 0.0

    def test_finding_one_of_many_is_not_a_success(self):
        # Hit rate would call this 1.0; that is the failure this set exists for.
        assert coverage([17, 1, 2, 3], {17, 5, 6, 33}, k=4) == 0.25

    def test_only_the_top_k_count(self):
        assert coverage([1, 2, 3, 17], {17}, k=3) == 0.0

    def test_duplicates_do_not_consume_budget(self):
        # Two chunks from Article 17 are one article, ranked where the first was.
        assert coverage([17, 17, 5], {17, 5}, k=2) == 1.0

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            coverage([17], {17}, k=0)

    def test_undefined_without_required_articles(self):
        with pytest.raises(ValueError, match="undefined"):
            coverage([17], set(), k=3)
