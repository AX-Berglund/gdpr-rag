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

from gdpr_rag.config import load_env
from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.evaluation import evaluate_retrieval, load_questions
from gdpr_rag.ingest import fixed_size_chunks, parse_document
from gdpr_rag.retrieve import Retriever
from gdpr_rag.rewrite import HyDERewriter, NullRewriter, PerspectiveRewriter, apply_rewrite

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def build_strategies(structured, sizes):
    """Name each chunking strategy alongside the chunks it produces."""
    strategies = {"structured": structured}
    for size in sizes:
        strategies[f"fixed-{size}"] = fixed_size_chunks(structured, size=size, overlap=size // 8)
    return strategies


def build_rewriters(names):
    """Resolve rewriter names, importing the model only when one is needed.

    Fails loudly on a missing key. ``apply_rewrite`` deliberately degrades to
    the original question when a rewriter errors, which is right at query time
    and disastrous in an experiment: every rewriting row would silently become
    a copy of the baseline and read as 'rewriting does nothing'.
    """
    import os

    if any(n != "none" for n in names) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Rewriting needs OPENAI_API_KEY (see .env.example). Without it every "
            "rewrite would fall back to the original question and the results "
            "would look like rewriting has no effect."
        )

    rewriters = []
    for name in names:
        if name == "none":
            rewriters.append(NullRewriter())
            continue
        from gdpr_rag.llm import OpenAIModel

        model = OpenAIModel()
        rewriters.append(HyDERewriter(model) if name == "hyde" else PerspectiveRewriter(model))
    return rewriters


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
    parser.add_argument(
        "--rewrite",
        nargs="+",
        default=["none"],
        choices=["none", "perspective", "hyde"],
        help="query rewriting strategies to compare (needs OPENAI_API_KEY beyond 'none')",
    )
    parser.add_argument(
        "--phrasing",
        nargs="+",
        default=["formal"],
        choices=["formal", "colloquial"],
        help="which question phrasing to score; pass both to measure the gap",
    )
    args = parser.parse_args()
    load_env()

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

    header = f"{'embedder':<34} {'chunking':<12} {'rewrite':<12} {'phrasing':<11} " + " ".join(
        f"{'hit@' + str(k):>7} {'mrr@' + str(k):>7} {'ndcg@' + str(k):>8}" for k in args.k
    )
    print(header)
    print("-" * len(header))

    rows = []
    perspective_reports = {}
    for embedder in build_embedders(args.dense):
        # Warm up so lazy model loading is not charged to the first strategy.
        embedder.encode(["warm-up"])
        for name, chunks in strategies.items():
            retriever = Retriever.build(embedder, chunks)
            for rewriter in build_rewriters(args.rewrite):
                # One rewrite per question, reused across every k.
                cache: dict[str, str] = {}

                def retrieve_fn(q, kk, r=retriever, rw=rewriter, c=cache):
                    if q not in c:
                        c[q] = apply_rewrite(rw, q)
                    return [x.chunk.article for x in r.retrieve(c[q], kk)]

                for phrasing in args.phrasing:
                    started = time.perf_counter()
                    cells = []
                    for k in args.k:
                        report = evaluate_retrieval(
                            questions,
                            retrieve_fn,
                            k=k,
                            embedder=embedder.name,
                            phrasing=phrasing,
                        )
                        cells.append(
                            f"{report.hit_rate:>7.3f} {report.mrr:>7.3f} {report.ndcg:>8.3f}"
                        )
                        rows.append({"chunking": name, **report.as_row()})
                        if name == "structured" and k == 5:
                            key = f"{embedder.name} / {rewriter.name} / {phrasing}"
                            perspective_reports[key] = report
                    elapsed = time.perf_counter() - started
                    print(
                        f"{embedder.name:<34} {name:<12} {rewriter.name[:11]:<12} {phrasing:<11} "
                        + " ".join(cells)
                        + f"   ({elapsed:.1f}s)"
                    )
            retriever.store.close()

    print(f"\nExcluded from means: {len(questions) - answerable} unanswerable questions.")

    # The headline mean hides the asymmetry that matters most: the regulation
    # is written as obligations on controllers, so questions asked from the
    # individual's side start further from the corpus register.
    if perspective_reports:
        print("\nhit@5 by who is asking (structured chunking):")
        for label, report in perspective_reports.items():
            split = ", ".join(f"{g} {v:.3f}" for g, v in report.by_perspective().items())
            counts = {
                g: sum(r.perspective == g for r in report.results)
                for g in ("subject", "organisation", "neutral")
            }
            print(f"  {label:<48} {split}")
        print("  n = " + ", ".join(f"{g} {c}" for g, c in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
