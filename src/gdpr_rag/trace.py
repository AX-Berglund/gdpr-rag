"""Recording what happened during a query.

A metric tells you the pipeline scored 0.306. It does not tell you whether the
decomposition was wrong, the retrieval missed a good sub-question, or the
generator ignored an article it was handed. Without that, a bad number is a
dead end.

So every stage records what it received and what it produced, nested to match
the shape of the work. That is all this is: a data structure and a stopwatch.
It is deliberately *not* a framework — there is no registry, no graph, no
chaining abstraction. Stages stay plain functions that happen to accept an
optional trace, so the library works identically without one.

    trace = Trace("can I get my data deleted?")
    with trace.span("retrieve", k=5) as span:
        results = ...
        span.record(articles=[r.chunk.article for r in results])
    print(trace.summary())
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Traces are read by humans and diffed between runs, so unbounded values make
# both useless. Generous enough that a model's answer survives intact, tight
# enough that a trace stays a page rather than a transcript.
MAX_VALUE_LENGTH = 2000


def _summarise(value: Any) -> Any:
    """Shrink a recorded value to something readable in a trace."""
    if isinstance(value, str):
        if len(value) <= MAX_VALUE_LENGTH:
            return value
        return f"{value[:MAX_VALUE_LENGTH]}… (+{len(value) - MAX_VALUE_LENGTH} chars)"
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return [_summarise(v) for v in value[:20]] + [f"… (+{len(value) - 20} more)"]
        return [_summarise(v) for v in value]
    if isinstance(value, dict):
        return {k: _summarise(v) for k, v in value.items()}
    return value


@dataclass
class Span:
    """One stage of work, and any stages nested inside it."""

    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    children: list[Span] = field(default_factory=list)

    def record(self, **outputs: Any) -> None:
        """Record what this stage produced. Callable more than once."""
        self.outputs.update({k: _summarise(v) for k, v in outputs.items()})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.inputs:
            data["inputs"] = self.inputs
        if self.outputs:
            data["outputs"] = self.outputs
        if self.error:
            data["error"] = self.error
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


class Trace:
    """Everything that happened while answering one query.

    Not thread-safe by design: a trace belongs to a single query on a single
    thread. Sharing one across threads would interleave the span stack and
    produce a tree that never happened.
    """

    def __init__(self, query: str) -> None:
        self.query = query
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **inputs: Any) -> Iterator[Span]:
        """Time a stage and attach it beneath whichever stage is running."""
        current = Span(name=name, inputs={k: _summarise(v) for k, v in inputs.items()})
        (self._stack[-1].children if self._stack else self.spans).append(current)
        self._stack.append(current)

        started = time.perf_counter()
        try:
            yield current
        except Exception as exc:
            # A failed stage is the most interesting thing in a trace, so it is
            # recorded before the exception continues on its way.
            current.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            current.duration_ms = (time.perf_counter() - started) * 1000
            self._stack.pop()

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    def find(self, name: str) -> list[Span]:
        """Every span with this name, at any depth."""

        def walk(spans: list[Span]) -> Iterator[Span]:
            for span in spans:
                yield span
                yield from walk(span.children)

        return [s for s in walk(self.spans) if s.name == name]

    def failed(self) -> list[Span]:
        """Spans that raised, at any depth."""

        def walk(spans: list[Span]) -> Iterator[Span]:
            for span in spans:
                yield span
                yield from walk(span.children)

        return [s for s in walk(self.spans) if s.error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "duration_ms": round(self.duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """A readable tree, for a terminal or a demo panel."""
        lines = [f'query: "{self.query}"  ({self.duration_ms:.0f} ms total)']

        def render(spans: list[Span], depth: int) -> None:
            for span in spans:
                marker = "✗" if span.error else "·"
                lines.append(f"{'  ' * (depth + 1)}{marker} {span.name}  {span.duration_ms:.0f} ms")
                for key, value in span.outputs.items():
                    lines.append(f"{'  ' * (depth + 2)}{key}: {value}")
                if span.error:
                    lines.append(f"{'  ' * (depth + 2)}error: {span.error}")
                render(span.children, depth + 1)

        render(self.spans, 0)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Trace({self.query!r}, {len(self.spans)} spans, {self.duration_ms:.0f} ms)"
