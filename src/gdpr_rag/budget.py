"""Capping what a public demo is willing to pay for.

A demo with an API key behind it spends its owner's money on behalf of
strangers. Retrieval — the part this project measures — costs nothing and stays
unlimited; only answer generation is capped.

Two limits, because they stop different things. A per-session cap stops one
curious visitor exploring at length. A daily cap stops everyone at once, which
is what a scraper or a shared link looks like. Neither replaces a spending
limit set with the API provider: this runs inside the process and cannot help
if the process is restarted or duplicated, so it is the second line of defence
rather than the first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

DEFAULT_PER_SESSION = 3
DEFAULT_PER_DAY = 150


@dataclass(frozen=True)
class Usage:
    """How much has been spent today. Resets when the date rolls over."""

    day: date
    count: int = 0

    def on(self, today: date) -> Usage:
        """This usage as it applies to ``today``, resetting across midnight."""
        return self if self.day == today else Usage(day=today, count=0)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class GenerationBudget:
    """How many answers a demo will generate before declining."""

    per_session: int = DEFAULT_PER_SESSION
    per_day: int = DEFAULT_PER_DAY

    def __post_init__(self) -> None:
        if self.per_session < 1 or self.per_day < 1:
            raise ValueError("limits must be at least 1")

    def check(self, usage: Usage, session_count: int, today: date) -> Decision:
        """Whether one more generation is allowed."""
        if session_count >= self.per_session:
            return Decision(
                False,
                f"This demo generates {self.per_session} answers per visit. "
                "Retrieval below is unlimited — it is also the part this project measures.",
            )
        if usage.on(today).count >= self.per_day:
            return Decision(
                False,
                "The demo has reached its daily generation limit. "
                "Retrieval below is unlimited and unaffected.",
            )
        return Decision(True)

    def spend(self, usage: Usage, today: date) -> Usage:
        """Record one generation against today's allowance."""
        current = usage.on(today)
        return replace(current, count=current.count + 1)
