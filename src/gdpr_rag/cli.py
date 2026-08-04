"""Command line interface.

Two commands, matching the two things you actually do: build an index from a
downloaded corpus, then ask it questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gdpr_rag.config import load_env
from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.generate import answer_question
from gdpr_rag.ingest import parse_document
from gdpr_rag.retrieve import Retriever
from gdpr_rag.store import ChunkStore
from gdpr_rag.trace import Trace

DEFAULT_INDEX = Path("data/index.sqlite")


def _embedder(name: str):
    """Resolve an embedder by name, importing torch only if actually needed."""
    if name == "hashing":
        return HashingEmbedder(dimensions=2048)
    from gdpr_rag.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder() if name == "local" else SentenceTransformerEmbedder(name)


def cmd_index(args: argparse.Namespace) -> int:
    source = args.source or next(iter(sorted(Path("data/raw").glob("*.html"))), None)
    if source is None:
        print("No corpus found. See the README for the download step.", file=sys.stderr)
        return 1

    chunks = parse_document(source)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    if args.index.exists():
        args.index.unlink()

    embedder = _embedder(args.embedder)
    print(f"Embedding {len(chunks)} chunks with {embedder.name}...")
    retriever = Retriever.build(embedder, chunks, path=str(args.index))
    retriever.store.close()

    articles = len({c.article for c in chunks})
    print(f"Indexed {len(chunks)} chunks across {articles} articles -> {args.index}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"No index at {args.index}. Run 'gdpr-rag index' first.", file=sys.stderr)
        return 1

    store = ChunkStore(args.index)
    embedder = _embedder(args.embedder)
    trace = Trace(args.question) if args.trace else None
    results = Retriever(embedder=embedder, store=store).retrieve(
        args.question, k=args.k, trace=trace
    )

    if args.answer:
        from gdpr_rag.llm import OpenAIModel

        answer = answer_question(args.question, results, OpenAIModel(), trace=trace)
        print(answer.text)
        print()
        if answer.unsupported_citations:
            print(f"⚠ unsupported citations: {', '.join(answer.unsupported_citations)}")
        print(f"grounded: {answer.is_grounded}")
        print()

    print(f"Retrieved for: {args.question}\n")
    for result in results:
        title = f" — {result.chunk.title}" if result.chunk.title else ""
        print(f"[{result.score:.3f}] {result.chunk.citation}{title}")
        print(f"        {result.chunk.text[:160]}\n")

    if trace is not None:
        print(trace.summary() if not args.trace_json else trace.to_json())
    store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gdpr-rag", description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--embedder",
        default="hashing",
        help="'hashing' (default, no download), 'local' (MiniLM), or a model name",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("index", help="build an index from a downloaded corpus")
    build.add_argument("source", type=Path, nargs="?", default=None)
    build.set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="retrieve articles for a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--answer", action="store_true", help="also generate a cited answer")
    ask.add_argument("--trace", action="store_true", help="show what each stage did")
    ask.add_argument("--trace-json", action="store_true", help="emit the trace as JSON")
    ask.set_defaults(func=cmd_ask)

    load_env()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
