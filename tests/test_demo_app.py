"""Behaviour of the public demo.

Everything here describes a fault that reached the deployed app. None of it
could have been caught locally, because a developer's environment has every
extra installed and a developer's session is always the first one.
"""

import importlib.util
import sys

from gdpr_rag.budget import GenerationBudget


class Slot:
    """Stand-in for a streamlit placeholder, remembering the last write."""

    def __init__(self):
        self.text = None

    def caption(self, text):
        self.text = text


class TestGenerationOffer:
    """Generation is offered on capability, not just on configuration.

    The client lives in an optional extra, so an environment can hold a valid
    key and still have no way to spend it. The deployed demo did exactly that:
    it enabled the toggle because a key was present, and raised ImportError the
    moment somebody used it.
    """

    def test_absent_client_means_generation_is_not_available(self, demo_namespace, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert demo_namespace["_generation_available"]() is False

    def test_present_client_means_generation_is_available(self, demo_namespace, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        assert demo_namespace["_generation_available"]() is True


class TestBudgetCaption:
    """The remaining allowance must count the answer on screen.

    The sidebar is drawn before the request is made, so reporting the count
    once, at that point, described the allowance as it stood before the visible
    answer existed — it read "3 of 3 left" beside the first of three. The count
    is rendered into a placeholder that is written again after the spend.
    """

    def test_reports_the_full_allowance_before_anything_is_spent(self, demo_namespace):
        slot = Slot()
        demo_namespace["_render_budget"](slot, GenerationBudget(per_session=3, per_day=100))
        assert slot.text == "3 of 3 left this visit."

    def test_rewriting_the_slot_after_a_spend_supersedes_the_first_write(self, demo_namespace):
        """The sequence the page actually performs, in order.

        The sidebar writes the count, a generation is then recorded, and the
        same placeholder is written again. What survives must describe the
        allowance after the spend, not before it — one placeholder written
        twice is what makes that possible.
        """
        slot = Slot()
        budget = GenerationBudget(per_session=3, per_day=100)
        state = sys.modules["streamlit"].session_state

        demo_namespace["_render_budget"](slot, budget)  # sidebar, before the request
        assert slot.text == "3 of 3 left this visit."

        state["generations"] = state.get("generations", 0) + 1  # the spend
        demo_namespace["_render_budget"](slot, budget)  # after the request
        assert slot.text == "2 of 3 left this visit."

    def test_never_reports_a_negative_allowance(self, demo_namespace):
        slot = Slot()
        sys.modules["streamlit"].session_state["generations"] = 9
        demo_namespace["_render_budget"](slot, GenerationBudget(per_session=3, per_day=100))
        assert slot.text == "0 of 3 left this visit."
