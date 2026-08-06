"""Tests for the demo generation budget."""

from datetime import date

import pytest

from gdpr_rag.budget import GenerationBudget, Usage

TODAY = date(2026, 8, 6)
TOMORROW = date(2026, 8, 7)


class TestSessionLimit:
    def test_allows_up_to_the_session_limit(self):
        budget = GenerationBudget(per_session=3, per_day=100)
        assert all(budget.check(Usage(TODAY), n, TODAY).allowed for n in range(3))

    def test_declines_beyond_it(self):
        budget = GenerationBudget(per_session=3, per_day=100)
        assert not budget.check(Usage(TODAY), 3, TODAY).allowed

    def test_the_reason_points_at_retrieval(self):
        # A refusal should tell the visitor what still works.
        budget = GenerationBudget(per_session=1, per_day=100)
        assert "Retrieval" in budget.check(Usage(TODAY), 1, TODAY).reason


class TestDailyLimit:
    def test_declines_once_the_day_is_spent(self):
        budget = GenerationBudget(per_session=10, per_day=5)
        assert not budget.check(Usage(TODAY, 5), 0, TODAY).allowed

    def test_a_new_day_resets_the_allowance(self):
        budget = GenerationBudget(per_session=10, per_day=5)
        assert budget.check(Usage(TODAY, 5), 0, TOMORROW).allowed

    def test_the_session_limit_binds_first(self):
        # Both exhausted: the visitor should hear the one they can act on.
        budget = GenerationBudget(per_session=1, per_day=1)
        assert "per visit" in budget.check(Usage(TODAY, 1), 1, TODAY).reason


class TestSpending:
    def test_recording_increments_today(self):
        assert GenerationBudget().spend(Usage(TODAY, 4), TODAY).count == 5

    def test_recording_on_a_new_day_starts_from_one(self):
        spent = GenerationBudget().spend(Usage(TODAY, 99), TOMORROW)
        assert spent == Usage(TOMORROW, 1)

    def test_usage_is_immutable(self):
        original = Usage(TODAY, 1)
        GenerationBudget().spend(original, TODAY)
        assert original.count == 1


class TestValidation:
    @pytest.mark.parametrize("kwargs", [{"per_session": 0}, {"per_day": 0}])
    def test_limits_below_one_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match="at least 1"):
            GenerationBudget(**kwargs)

    def test_defaults_are_conservative(self):
        budget = GenerationBudget()
        assert budget.per_session <= 5
        assert budget.per_day <= 500
