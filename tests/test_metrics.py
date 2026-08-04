"""Tests for retrieval metrics."""

import pytest

from gdpr_rag.evaluation.metrics import hit_rate, ndcg, reciprocal_rank, to_article_ranking


class TestArticleRanking:
    def test_duplicate_articles_collapse_to_first_appearance(self):
        # Three chunks from Article 17 are one hit, at the rank of the earliest.
        assert to_article_ranking([17, 17, 5, 17, 9]) == [17, 5, 9]

    def test_empty_ranking_is_empty(self):
        assert to_article_ranking([]) == []


class TestHitRate:
    def test_hit_within_k(self):
        assert hit_rate([5, 17, 9], {17}, k=3) == 1.0

    def test_miss_outside_k(self):
        assert hit_rate([5, 9, 17], {17}, k=2) == 0.0

    def test_any_labelled_article_counts(self):
        assert hit_rate([9, 33], {17, 33}, k=2) == 1.0

    def test_duplicates_do_not_push_a_hit_out_of_range(self):
        # Chunk-level this would be rank 3; article-level it is rank 2.
        assert hit_rate([5, 5, 17], {17}, k=2) == 1.0

    def test_no_hit_anywhere(self):
        assert hit_rate([1, 2, 3], {17}, k=3) == 0.0

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            hit_rate([17], {17}, k=0)

    def test_undefined_without_labels(self):
        with pytest.raises(ValueError, match="undefined"):
            hit_rate([17], set(), k=3)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "retrieved,expected",
        [([17, 5, 9], 1.0), ([5, 17, 9], 0.5), ([5, 9, 17], 1 / 3), ([1, 2, 3], 0.0)],
    )
    def test_rank_positions(self, retrieved, expected):
        assert reciprocal_rank(retrieved, {17}) == pytest.approx(expected)

    def test_uses_the_earliest_labelled_article(self):
        assert reciprocal_rank([5, 33, 17], {17, 33}) == pytest.approx(0.5)

    def test_duplicates_do_not_depress_the_rank(self):
        assert reciprocal_rank([5, 5, 5, 17], {17}) == pytest.approx(0.5)

    def test_undefined_without_labels(self):
        with pytest.raises(ValueError, match="undefined"):
            reciprocal_rank([17], set())


class TestNdcg:
    def test_perfect_ranking_scores_one(self):
        assert ndcg([17, 33, 5], {17, 33}, k=3) == pytest.approx(1.0)

    def test_no_relevant_articles_scores_zero(self):
        assert ndcg([1, 2, 3], {17}, k=3) == 0.0

    def test_higher_placement_scores_better(self):
        assert ndcg([5, 17], {17}, k=2) < ndcg([17, 5], {17}, k=2)

    def test_two_of_three_beats_one_of_three(self):
        assert ndcg([17, 33, 1], {17, 33, 44}, k=3) > ndcg([17, 1, 2], {17, 33, 44}, k=3)

    def test_more_labels_than_k_is_not_penalised_for_the_impossible(self):
        # Only two slots exist, so retrieving two correct articles is ideal.
        assert ndcg([17, 33], {17, 33, 44}, k=2) == pytest.approx(1.0)

    def test_k_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            ndcg([17], {17}, k=0)

    def test_undefined_without_labels(self):
        with pytest.raises(ValueError, match="undefined"):
            ndcg([17], set(), k=3)
