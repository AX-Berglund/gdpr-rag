"""Report retrieval quality across all three evaluation sets.

They are reported separately and never averaged, because they measure
different things and a combined number would mean nothing:

  questions  64 lookups, labels written by hand. The easiest set, and the only
             one where a single article answers the question.
  scenarios  18 narratives, labels written by hand. Several articles each,
             scored on coverage.
  cases      70 Court of Justice judgments, labels supplied by the Court. The
             only externally-labelled set, and the hardest — these reached the
             Court precisely because the law was unclear.

    python scripts/evaluate.py --dense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

from gdpr_rag.config import load_env
from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.evaluation import (
    coverage,
    evaluate_cases,
    evaluate_retrieval,
    load_cases,
    load_questions,
    load_scenarios,
)
from gdpr_rag.ingest import parse_document
from gdpr_rag.retrieve import Retriever

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--dense", action="store_true", help="use the local MiniLM model")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()
    load_env()

    html = args.html or next(iter(sorted(DATA.glob("*.html"))), None)
    if html is None:
        print(f"No corpus HTML in {DATA}. See the README.", file=sys.stderr)
        return 1

    if args.dense:
        from gdpr_rag.embed import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    else:
        embedder = HashingEmbedder(dimensions=2048)

    retriever = Retriever.build(embedder, parse_document(html))
    articles_for = lambda q, k: [x.chunk.article for x in retriever.retrieve(q, k)]  # noqa: E731

    questions, scenarios, cases = load_questions(), load_scenarios(), load_cases()
    print(f"retriever   {embedder.name}")
    print(f"corpus      {html.name}\n")

    print("QUESTIONS — 64 lookups, hand-labelled, one article each")
    print(f"  {'k':>3}  {'hit rate':>9}  {'MRR':>7}  {'nDCG':>7}   by perspective")
    for k in args.k:
        report = evaluate_retrieval(questions, articles_for, k=k, embedder=embedder.name)
        split = " ".join(f"{g}={v:.2f}" for g, v in report.by_perspective().items())
        print(
            f"  {k:>3}  {report.hit_rate:>9.3f}  {report.mrr:>7.3f}  {report.ndcg:>7.3f}   {split}"
        )
    print(f"  excluded: {report.excluded_unanswerable} unanswerable\n")

    print("SCENARIOS — 18 narratives, hand-labelled, several articles each")
    print(f"  {'k':>3}  {'coverage':>9}  {'found any':>10}  {'found all':>10}")
    for k in args.k:
        scores = [coverage(articles_for(s.scenario, k), s.article_numbers, k) for s in scenarios]
        print(
            f"  {k:>3}  {mean(scores):>9.3f}  {mean(c > 0 for c in scores):>10.3f}  "
            f"{mean(c == 1.0 for c in scores):>10.3f}"
        )
    print(f"  n={len(scenarios)}\n")

    print("CASES — 70 CJEU judgments, labelled by the Court, queries redacted")
    print(f"  {'k':>3}  {'coverage':>9}  {'found any':>10}  {'found all':>10}   by label source")
    for k in args.k:
        report = evaluate_cases(cases, articles_for, k=k, retriever=embedder.name)
        split = " ".join(f"{s}={v:.2f}" for s, v in report.by_label_source.items())
        print(
            f"  {k:>3}  {report.coverage:>9.3f}  {report.found_any:>10.3f}  "
            f"{report.found_all:>10.3f}   {split}"
        )
    raw = evaluate_cases(cases, articles_for, k=args.k[-1], redacted=False)
    print(f"  n={len(report.coverages)}, excluded {report.excluded} (procedural or unparsed)")
    print(
        f"  leak check @k={args.k[-1]}: redacted {report.coverage:.3f} vs "
        f"raw {raw.coverage:.3f} (questions give away {report.mean_leak:.0%} of labels)"
    )
    print(f"  worst: {', '.join(cid for cid, _ in report.worst(4))}")

    print("\nThe three sets are not comparable and are never averaged.")
    retriever.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
