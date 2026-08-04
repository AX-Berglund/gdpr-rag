"""Structure-aware chunking of regulation text.

The naive approach to RAG chunking is a fixed-size sliding window. That works
until a retrieved chunk straddles two paragraphs and the answer cites the wrong
one. The regulation already has an authoritative structure -- articles,
numbered paragraphs, lettered points, definitions -- so this module parses that
structure instead of ignoring it. Every chunk is therefore citable by
construction.

Whether it also *retrieves* better than fixed-size chunking is an empirical
question, and the evaluation harness answers it rather than assuming.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from gdpr_rag.ingest.models import Chunk, ChunkKind

# "Article 17" on a line of its own, which is how article headers appear in the
# official consolidated text.
_ARTICLE_HEADER = re.compile(r"^Article\s+(\d+)\s*$", re.MULTILINE)

# A numbered paragraph: "1. The data subject shall..."
_PARAGRAPH = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

# A lettered point inside a paragraph: "(a) the personal data are..."
_POINT = re.compile(r"^\(([a-z])\)\s+", re.MULTILINE)

# A numbered definition inside Article 4: "(7) 'controller' means..."
_DEFINITION = re.compile(r"^\((\d+)\)\s+", re.MULTILINE)


def split_articles(text: str) -> Iterator[tuple[int, str]]:
    """Split a full regulation document into ``(article_number, article_text)``.

    Text before the first article header (recitals, preamble) is skipped: it is
    not citable as an article and would otherwise be attributed to Article 1.
    """
    matches = list(_ARTICLE_HEADER.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield int(match.group(1)), text[match.start() : end].strip()


def classify(article_text: str) -> ChunkKind:
    """Determine which of the three article shapes this text has.

    Definitions are checked first because Article 4 contains ``(1)``-style
    entries that the paragraph rule would otherwise miss entirely.
    """
    body = _strip_header(article_text)[1]

    if _DEFINITION.search(body) and "definition" in _title(article_text).lower():
        return ChunkKind.DEFINITION
    if _PARAGRAPH.search(body):
        return ChunkKind.PARAGRAPH
    return ChunkKind.BODY


def parse_article(article_text: str) -> list[Chunk]:
    """Parse one article into its citable chunks."""
    number = _article_number(article_text)
    title = _title(article_text)
    kind = classify(article_text)
    _, body = _strip_header(article_text)

    if kind is ChunkKind.DEFINITION:
        return _parse_definitions(number, title, body)
    if kind is ChunkKind.PARAGRAPH:
        return _parse_paragraphs(number, title, body)
    return _parse_body(number, title, body)


def chunk_document(text: str) -> list[Chunk]:
    """Parse a full regulation document into citable chunks."""
    return [chunk for _, article in split_articles(text) for chunk in parse_article(article)]


def _article_number(article_text: str) -> int:
    match = _ARTICLE_HEADER.search(article_text)
    if match is None:
        raise ValueError(f"No article header found in: {article_text[:80]!r}")
    return int(match.group(1))


def _title(article_text: str) -> str:
    """The article title, which sits on the line after the header.

    Returns an empty string for articles that have no title line.
    """
    lines = [line.strip() for line in article_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return ""
    # A second line that already starts the substance is not a title.
    second = lines[1]
    if _PARAGRAPH.match(second) or _DEFINITION.match(second):
        return ""
    return second


def _strip_header(article_text: str) -> tuple[str, str]:
    """Split an article into ``(header, body)``, dropping the title line."""
    lines = article_text.splitlines()
    header = lines[0] if lines else ""
    rest = lines[1:]
    if rest and _title(article_text):
        # Drop the title line, wherever the leading blank lines put it.
        for i, line in enumerate(rest):
            if line.strip():
                rest = rest[i + 1 :]
                break
    return header, "\n".join(rest).strip()


def _split_on(pattern: re.Pattern[str], text: str) -> list[tuple[str, str]]:
    """Split ``text`` on ``pattern``, returning ``(label, segment)`` pairs.

    Text preceding the first match is returned with an empty label, so callers
    can decide whether a lead-in is worth keeping.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text.strip())] if text.strip() else []

    segments: list[tuple[str, str]] = []
    lead_in = text[: matches[0].start()].strip()
    if lead_in:
        segments.append(("", lead_in))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[match.end() : end].strip()
        if segment:
            segments.append((match.group(1), segment))
    return segments


def _parse_paragraphs(number: int, title: str, body: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for para_label, para_text in _split_on(_PARAGRAPH, body):
        points = _split_on(_POINT, para_text)
        # A paragraph with lettered points becomes one chunk for its lead-in
        # sentence plus one per point, because the points are independently
        # citable conditions.
        for point_label, point_text in points:
            chunks.append(
                Chunk(
                    article=number,
                    kind=ChunkKind.PARAGRAPH,
                    text=point_text,
                    title=title or None,
                    paragraph=para_label or None,
                    point=point_label or None,
                )
            )
    return chunks


def _parse_definitions(number: int, title: str, body: str) -> list[Chunk]:
    return [
        Chunk(
            article=number,
            kind=ChunkKind.DEFINITION,
            text=text,
            title=title or None,
            definition=label,
        )
        for label, text in _split_on(_DEFINITION, body)
        if label
    ]


def _parse_body(number: int, title: str, body: str) -> list[Chunk]:
    text = " ".join(line.strip() for line in body.splitlines() if line.strip())
    if not text:
        return []
    return [
        Chunk(
            article=number,
            kind=ChunkKind.BODY,
            text=text,
            title=title or None,
        )
    ]
