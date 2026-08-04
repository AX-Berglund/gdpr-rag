"""Check a downloaded corpus against what the parser produces.

The unit tests run against a small fixture, so nothing in CI proves the parser
handles all 99 articles of the real regulation. This script does, and it is the
gate to run after downloading the corpus and after any change to the parser.

Where both sources are present it also cross-validates them: the HTML and the
PDF are independent renderings of the same regulation, so if they disagree on
which articles exist, one of the two parsers is wrong.

    python scripts/validate_corpus.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from gdpr_rag.ingest.eurlex import parse_document

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED_ARTICLES = 99

# Official Journal page furniture. Legitimate in three articles that genuinely
# refer to publication in the Journal; anywhere else it is extraction debris.
_FURNITURE = re.compile(r"Official Journal|L \d+/\d+")
_FURNITURE_EXPECTED = {45, 92, 99}


def _find(pattern: str) -> Path | None:
    return next(iter(sorted(DATA.glob(pattern))), None)


def validate_html(path: Path) -> tuple[bool, set[int]]:
    chunks = parse_document(path)
    articles = {c.article for c in chunks}
    missing = sorted(set(range(1, EXPECTED_ARTICLES + 1)) - articles)
    kinds = Counter(c.kind.value for c in chunks)

    print(f"HTML  {path.name}")
    print(f"      {len(chunks)} chunks across {len(articles)} articles — {dict(kinds)}")

    ok = True
    if missing:
        print(f"      FAIL missing articles: {missing}")
        ok = False
    if empty := [c.citation for c in chunks if not c.text.strip()]:
        print(f"      FAIL {len(empty)} empty chunks: {empty[:5]}")
        ok = False
    if dirty := [c for c in chunks if "\n" in c.text or "  " in c.text]:
        print(f"      FAIL {len(dirty)} chunks with unnormalised whitespace")
        ok = False

    furniture = {c.article for c in chunks if _FURNITURE.search(c.text)}
    if unexpected := furniture - _FURNITURE_EXPECTED:
        print(f"      FAIL page furniture leaked into articles {sorted(unexpected)}")
        ok = False

    if ok:
        print("      OK")
    return ok, articles


def validate_pdf(path: Path, html_articles: set[int]) -> bool:
    """Cross-check the PDF against the HTML, if the optional extra is installed."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print(f"PDF   {path.name}\n      skipped (pip install 'gdpr-rag[pdf]')")
        return True

    from gdpr_rag.ingest.chunk import chunk_document

    text = extract_text(str(path))
    # Strip the running header before chunking; unlike the HTML, the PDF
    # interleaves page furniture with the body text.
    text = re.sub(
        r"\n?\d+\.\d+\.\d{4}\s*\n+EN\s*\n+Official\s+Journal[^\n]*\n+L\s*\d+/\d+\s*\n?",
        "\n",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    articles = {c.article for c in chunk_document(text)}

    print(f"PDF   {path.name}")
    print(f"      {len(articles)} articles")
    if articles == html_articles:
        print("      OK — agrees with HTML")
        return True
    print(
        f"      FAIL disagrees with HTML: only-HTML={sorted(html_articles - articles)} "
        f"only-PDF={sorted(articles - html_articles)}"
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()

    html = args.html or _find("*.html")
    if html is None:
        print(f"No HTML corpus in {DATA}. See the README for the download step.")
        return 1

    ok, articles = validate_html(html)

    pdf = args.pdf or _find("*.pdf")
    if pdf is not None:
        ok = validate_pdf(pdf, articles) and ok
    else:
        print("PDF   none found — cross-validation skipped")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
