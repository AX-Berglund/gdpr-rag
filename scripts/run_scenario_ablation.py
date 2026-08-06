"""Measure query strategies on the narrative set.

The questions set asks about a rule and scores 0.700 hit@5. The scenarios set
describes a situation and asks whether it was allowed, and scores 0.306
coverage@5. That gap is the open problem, and this measures what closes it.

Four strategies, all at the same evidence budget so the comparison is fair —
each returns k chunks, whatever it cost to find them:

  none          the question as typed
  perspective   restated in the regulation's obligation-side vocabulary
  decompose     split into sub-questions, retrieved separately, rank-fused
  both          the union of the two query sets, rank-fused

Costs are reported alongside, because a strategy that wins by making four
model calls per question is a different proposition from one that makes one.

    python scripts/run_scenario_ablation.py                    # scenarios
    python scripts/run_scenario_ablation.py --set questions    # control
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gdpr_rag.config import load_env
from gdpr_rag.evaluation import load_questions, load_scenarios
from gdpr_rag.evaluation.scenarios import coverage
from gdpr_rag.ingest import parse_document
from gdpr_rag.multiquery import Decomposer, multi_query_retrieve
from gdpr_rag.retrieve import Retriever
from gdpr_rag.rewrite import HyDERewriter, PerspectiveRewriter

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


class Strategy:
    """A named way of turning one question into the queries actually issued."""

    def __init__(self, name, expand, calls):
        self.name = name
        self.expand = expand
        self.calls = calls


def build_strategies(model):
    """Every strategy, including the one the README currently recommends.

    HyDE is in here because a new method that is not compared against the
    incumbent is not a result. Fusions are named for what they combine.
    """
    perspective = PerspectiveRewriter(model)
    hyde = HyDERewriter(model)
    decomposer = Decomposer(model)

    return [
        Strategy("none", lambda q: [q], 0),
        Strategy("hyde", lambda q: [hyde.rewrite(q)], 1),
        Strategy("perspective", lambda q: [perspective.rewrite(q)], 1),
        Strategy("decompose", decomposer.decompose, 1),
        Strategy("decomp+persp", lambda q: decomposer.decompose(q) + [perspective.rewrite(q)], 2),
        Strategy("decomp+hyde", lambda q: decomposer.decompose(q) + [hyde.rewrite(q)], 2),
    ]


def load_items(name):
    """Return (text, required articles) pairs for the chosen set."""
    if name == "scenarios":
        return [(s.scenario, s.article_numbers) for s in load_scenarios()]
    # Questions are scored by the same coverage metric so the two sets stay
    # comparable; over the single-article label a question carries, coverage is
    # hit rate. The colloquial phrasing is used because that is what a user
    # types, and it is the phrasing rewriting exists to repair.
    return [
        (q.phrased("colloquial"), q.article_numbers) for q in load_questions() if not q.unanswerable
    ]


def evaluate(retriever, items, strategy, ks):
    """Coverage at each k, plus wall-clock, for one strategy."""
    totals = {k: 0.0 for k in ks}
    started = time.perf_counter()
    for text, required in items:
        queries = strategy.expand(text)
        for k in ks:
            results = multi_query_retrieve(retriever, queries, k=k)
            totals[k] += coverage([r.chunk.article for r in results], required, k)
    elapsed = time.perf_counter() - started
    return {k: totals[k] / len(items) for k in ks}, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="scenarios", choices=["scenarios", "questions"])
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--limit", type=int, help="evaluate only the first N items")
    args = parser.parse_args()

    load_env()
    corpus = next(iter(sorted(DATA.glob("*.html"))), None)
    if corpus is None:
        print(f"No corpus in {DATA}. See the README.", file=sys.stderr)
        return 1

    from gdpr_rag.embed import SentenceTransformerEmbedder
    from gdpr_rag.llm import OpenAIModel

    items = load_items(args.set)
    if args.limit:
        items = items[: args.limit]

    retriever = Retriever.build(SentenceTransformerEmbedder(), parse_document(str(corpus)))
    model = OpenAIModel()

    print(f"\n{args.set}: {len(items)} items\n")
    header = "  ".join(f"cov@{k}" for k in args.k)
    print(f"{'strategy':<14} {header}  {'calls':>6}  {'secs':>6}")
    print("-" * (14 + len(header) + 18))

    baseline = None
    for strategy in build_strategies(model):
        scores, elapsed = evaluate(retriever, items, strategy, args.k)
        if baseline is None:
            baseline = scores
        cells = []
        for k in args.k:
            delta = scores[k] - baseline[k]
            cells.append(f"{scores[k]:.3f}{'' if baseline is scores else f' ({delta:+.3f})'}")
        calls = strategy.calls * len(items)
        print(f"{strategy.name:<14} {'  '.join(cells)}  {calls:>6}  {elapsed:>6.1f}")

    print("\nEvery strategy returns the same number of chunks; only the queries differ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
