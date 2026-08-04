"""Measure retrieval quality across chunking strategies and embedders.

This is the experiment the project exists to run. Everything else is
infrastructure for producing this table honestly:

  - the same corpus text feeds every configuration, so differences come from
    the variable under test rather than from extraction quality
  - the same labelled questions score every configuration
  - unanswerable questions are excluded from retrieval means, and the count of
    exclusions is printed rather than hidden

    python scripts/run_ablation.py                 # hashing baseline only, seconds
    python scripts/run_ablation.py --dense         # add the local MiniLM model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.evaluation import evaluate_retrieval, load_questions
from gdpr_rag.ingest import fixed_size_chunks, parse_document
from gdpr_rag.retrieve import Retriever

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def build_strategies(structured, sizes):
    """Name each chunking strategy alongside the chunks it produces."""
    strategies = {"structured": structured}
    for size in sizes:
        strategies[f"fixed-{size}"] = fixed_size_chunks(structured, size=size, overlap=size // 8)
    return strategies


def build_embedders(dense: bool):
    embedders = [HashingEmbedder(dimensions=2048)]
    if dense:
        from gdpr_rag.embed import SentenceTransformerEmbedder

        embedders.append(SentenceTransformerEmbedder())
    return embedders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--dense", action="store_true", help="include the local MiniLM model")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--sizes", type=int, nargs="+", default=[400, 800, 1600])
    args = parser.parse_args()

    html = args.html or next(iter(sorted(DATA.glob("*.html"))), None)
    if html is None:
        print(f"No corpus HTML in {DATA}. See the README for the download step.")
        return 1

    structured = parse_document(html)
    strategies = build_strategies(structured, args.sizes)
    questions = load_questions()
    answerable = sum(not q.unanswerable for q in questions)

    print(f"corpus     {html.name}")
    print(f"questions  {answerable} answerable, {len(questions) - answerable} refusal probes")
    print("chunks     " + ", ".join(f"{n}={len(c)}" for n, c in strategies.items()))
    print()

    header = f"{'embedder':<34} {'chunking':<12} " + " ".join(
        f"{'hit@' + str(k):>7} {'mrr@' + str(k):>7} {'ndcg@' + str(k):>8}" for k in args.k
    )
    print(header)
    print("-" * len(header))

    rows = []
    for embedder in build_embedders(args.dense):
        # Warm up so lazy model loading is not charged to the first strategy.
        embedder.encode(["warm-up"])
        for name, chunks in strategies.items():
            started = time.perf_counter()
            retriever = Retriever.build(embedder, chunks)
            cells = []
            for k in args.k:
                report = evaluate_retrieval(
                    questions,
                    lambda q, kk, r=retriever: [x.chunk.article for x in r.retrieve(q, kk)],
                    k=k,
                    embedder=embedder.name,
                )
                cells.append(f"{report.hit_rate:>7.3f} {report.mrr:>7.3f} {report.ndcg:>8.3f}")
                rows.append({"chunking": name, **report.as_row()})
            retriever.store.close()
            elapsed = time.perf_counter() - started
            print(f"{embedder.name:<34} {name:<12} " + " ".join(cells) + f"   ({elapsed:.1f}s)")

    print(f"\nExcluded from means: {len(questions) - answerable} unanswerable questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
