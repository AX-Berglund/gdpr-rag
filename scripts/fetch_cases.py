"""Fetch Court of Justice judgments and extract evaluation labels.

Reads a list of case numbers, downloads each judgment from EUR-Lex, and writes
a manifest of what could be read out of them.

Judgments are not vendored — they land in a gitignored directory and the repo
keeps only the manifest, which is small, auditable and free of licensing
questions.

    python scripts/fetch_cases.py --discover      # ask the EU triplestore
    python scripts/fetch_cases.py cases.txt       # or read a hand-made list

Discovery uses the Publications Office SPARQL endpoint, which records which
judgments interpret which legal act. That is the Court's own answer to "is this
a GDPR case", so the case list is externally determined rather than assembled
by whoever happened to remember some case numbers.

A hand-made list may hold either form, one per line, with `#` for comments:

    C-300/21
    62021CJ0300
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from gdpr_rag.cases.extract import NotAGdprCase, NotAJudgment, extract

SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"

# cdm:case-law_interpretes_resource_legal is the Publications Office's own
# record of which judgments interpret which act.
DISCOVERY_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex WHERE {
  ?act cdm:resource_legal_id_celex ?id . FILTER(STR(?id) = "32016R0679")
  ?case cdm:case-law_interpretes_resource_legal ?act .
  ?case cdm:resource_legal_id_celex ?celex .
} ORDER BY ?celex
"""

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cases"
MANIFEST = ROOT / "evaluation" / "cases.yaml"
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

# "C-300/21" — the year is when the case was lodged, which is what CELEX uses.
_CASE_NUMBER = re.compile(r"^([CTF])-(\d{1,4})/(\d{2})$", re.IGNORECASE)
_CELEX = re.compile(r"^6\d{4}[A-Z]{2}\d{4}$", re.IGNORECASE)


def to_celex(reference: str) -> str:
    """Convert ``C-300/21`` to ``62021CJ0300``. CELEX ids pass through."""
    reference = reference.strip()
    if _CELEX.match(reference):
        return reference.upper()

    match = _CASE_NUMBER.match(reference)
    if not match:
        raise ValueError(f"unrecognised case reference {reference!r} (expected C-300/21)")
    court, number, year = match.groups()
    # Two-digit years: the Court has existed since 1953, so anything above the
    # current decade would be last century — but GDPR cases are all post-2018.
    full_year = 2000 + int(year) if int(year) < 90 else 1900 + int(year)
    return f"6{full_year}{court.upper()}J{int(number):04d}"


def discover() -> list[str]:
    """Ask the EU triplestore which judgments interpret the GDPR."""
    import json
    import urllib.parse

    query = urllib.parse.urlencode(
        {"query": DISCOVERY_QUERY, "format": "application/sparql-results+json"}
    )
    request = urllib.request.Request(f"{SPARQL}?{query}", headers={"User-Agent": "gdpr-rag"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    return [b["celex"]["value"] for b in payload["results"]["bindings"]]


def read_list(path: Path) -> list[str]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            entries.append(line)
    return entries


def download(celex: str, *, force: bool = False) -> str:
    """Fetch a judgment, caching it so re-runs do not hammer EUR-Lex."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{celex}.html"
    if path.exists() and not force:
        return path.read_text(encoding="utf-8", errors="replace")

    request = urllib.request.Request(URL.format(celex=celex), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
    path.write_bytes(body)
    time.sleep(1)  # EUR-Lex is a public service; do not hammer it.
    return body.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("list", type=Path, nargs="?", help="file of case references, one per line")
    parser.add_argument(
        "--discover", action="store_true", help="ask the EU triplestore for the case list"
    )
    parser.add_argument("--out", type=Path, default=MANIFEST)
    parser.add_argument("--force", action="store_true", help="re-download cached judgments")
    args = parser.parse_args()

    if args.discover:
        print("Asking the EU triplestore which judgments interpret the GDPR...")
        entries = discover()
    elif args.list and args.list.exists():
        entries = read_list(args.list)
    else:
        print("Pass --discover, or a case list file. See the README.", file=sys.stderr)
        return 1

    records, skipped = [], []
    print(f"{len(entries)} references\n")

    for reference in entries:
        try:
            celex = to_celex(reference)
        except ValueError as exc:
            skipped.append((reference, str(exc)))
            print(f"  {reference:<14} SKIP  {exc}")
            continue

        try:
            raw = download(celex, force=args.force)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            skipped.append((reference, f"download failed: {exc}"))
            print(f"  {reference:<14} FAIL  download: {exc}")
            continue

        try:
            found = extract(celex, raw)
        except NotAJudgment as exc:
            # Transport problem, not a content one — worth retrying later.
            skipped.append((reference, str(exc)))
            print(f"  {reference:<14} FAIL  no decision text (retry with --force)")
            continue
        except NotAGdprCase as exc:
            # Expected and useful: a list assembled by hand will contain cases
            # that turned out not to be about the GDPR at all.
            skipped.append((reference, str(exc)))
            print(f"  {reference:<14} SKIP  not a GDPR case")
            continue

        records.append(
            {
                "id": reference,
                "celex": celex,
                "court": "CJEU",
                "title": found.title,
                "articles": [f"Article {n}" for n in found.articles],
                "label_source": found.label_source,
                "mentions": found.mentions,
                "questions": found.questions,
                "source": URL.format(celex=celex).replace("/TXT/HTML/", "/TXT/"),
            }
        )
        print(f"  {reference:<14} ok    {found.articles} via {found.label_source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Generated by scripts/fetch_cases.py — labels read from the judgments\n"
        "# themselves, not written by hand. Re-run to regenerate.\n"
        + yaml.safe_dump(records, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )

    # A case can legitimately have no substantive articles: some concern only
    # the supervisory authority's competence or tasks, which are procedural.
    unlabelled = [r["id"] for r in records if not r["articles"]]
    print(f"\n{len(records)} cases -> {args.out.relative_to(ROOT)}")
    print(f"{len(skipped)} skipped")
    if unlabelled:
        print(
            f"{len(unlabelled)} with no substantive articles (procedural cases, "
            f"excluded from evaluation): {unlabelled}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
