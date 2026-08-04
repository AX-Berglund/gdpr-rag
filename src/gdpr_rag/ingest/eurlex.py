"""Parsing the EUR-Lex Official Journal XHTML rendering of a regulation.

This is the preferred ingest path. The PDF carries the same text but not the
same information: in the PDF an article header is a line that *looks* like a
header and has to be told apart from the hundreds of cross-references reading
"Article 6(1)", page furniture bleeds into the body at every page break, and
justified typesetting leaves doubled spaces and mid-sentence line breaks.

The Official Journal HTML marks all of it explicitly:

    <div class="eli-subdivision" id="art_17">      <- the article, numbered
      <p class="oj-ti-art">Article 17</p>
      <div class="eli-title"><p class="oj-sti-art">Right to erasure…</p></div>
      <div id="017.001">                           <- paragraph 1
        <p class="oj-normal">1.  The data subject…</p>
        <table>…<td>(a)</td><td>the personal data…</td>…</table>   <- point (a)

So structure is read rather than inferred, and there is no page furniture to
strip because headers and footers are print artifacts that do not exist here.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import html as lxml_html

from gdpr_rag.ingest.models import Chunk, ChunkKind

# "art_17" -> 17. Annexes and recitals use other id prefixes and are skipped.
_ARTICLE_ID = re.compile(r"^art_(\d+)$")

# Paragraph containers are "017.001": article, then paragraph.
_PARAGRAPH_ID = re.compile(r"^(\d+)\.(\d+)$")

# Paragraph text repeats its own number ("1.  The data subject…"). The
# structural id is authoritative, so the prefix is redundant.
_LEADING_NUMBER = re.compile(r"^\d+\.\s*")

# Table labels: "(a)" for points, "(7)" for definitions.
_LABEL = re.compile(r"^\(([a-z0-9]+)\)$", re.IGNORECASE)


def _clean(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces EUR-Lex uses."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _strip_footnotes(element) -> None:
    """Remove footnote markers and bodies in place.

    A superscript footnote tag left in the text becomes a stray digit glued to
    a word, which then lands in the embedding as noise.
    """
    for note in element.xpath('.//*[contains(@class, "oj-note")]'):
        parent = note.getparent()
        if parent is not None:
            # Keep any trailing text so neighbouring words do not run together.
            tail = note.tail or ""
            previous = note.getprevious()
            if previous is not None:
                previous.tail = (previous.tail or "") + tail
            else:
                parent.text = (parent.text or "") + tail
            parent.remove(note)


def _labelled_rows(container) -> list[tuple[str, str]]:
    """Extract ``(label, text)`` pairs from the two-cell tables EUR-Lex uses.

    Points and definitions share this layout: a narrow cell holding "(a)" or
    "(7)", and a wide cell holding the text.
    """
    rows = []
    for table in container.xpath("./table"):
        cells = table.xpath(".//td")
        if len(cells) < 2:
            continue
        label = _LABEL.match(_clean(cells[0].text_content()))
        text = _clean(cells[1].text_content())
        if label and text:
            rows.append((label.group(1), text))
    return rows


def _article_number(div) -> int | None:
    match = _ARTICLE_ID.match(div.get("id") or "")
    return int(match.group(1)) if match else None


def _title(div) -> str | None:
    found = div.xpath('.//p[@class="oj-sti-art"]')
    return _clean(found[0].text_content()) if found else None


def _parse_paragraphs(div, number: int, title: str | None) -> list[Chunk]:
    """Articles built from numbered paragraphs, each optionally with points."""
    chunks: list[Chunk] = []
    for container in div.xpath("./div[@id]"):
        match = _PARAGRAPH_ID.match(container.get("id") or "")
        if not match:
            continue
        paragraph = str(int(match.group(2)))

        for element in container.xpath('./p[@class="oj-normal"]'):
            text = _LEADING_NUMBER.sub("", _clean(element.text_content()))
            if text:
                chunks.append(
                    Chunk(
                        article=number,
                        kind=ChunkKind.PARAGRAPH,
                        text=text,
                        title=title,
                        paragraph=paragraph,
                    )
                )

        for label, text in _labelled_rows(container):
            chunks.append(
                Chunk(
                    article=number,
                    kind=ChunkKind.PARAGRAPH,
                    text=text,
                    title=title,
                    paragraph=paragraph,
                    point=label.lower(),
                )
            )
    return chunks


def _parse_definitions(div, number: int, title: str | None) -> list[Chunk]:
    """Article 4, whose numbered entries are definitions rather than paragraphs."""
    return [
        Chunk(
            article=number,
            kind=ChunkKind.DEFINITION,
            text=text,
            title=title,
            definition=label,
        )
        for label, text in _labelled_rows(div)
    ]


def _parse_body(div, number: int, title: str | None) -> list[Chunk]:
    """Articles that are plain prose, with no numbered subdivision."""
    chunks = [
        Chunk(article=number, kind=ChunkKind.BODY, text=text, title=title)
        for text in (_clean(p.text_content()) for p in div.xpath('./p[@class="oj-normal"]'))
        if text
    ]
    chunks.extend(
        Chunk(
            article=number,
            kind=ChunkKind.PARAGRAPH,
            text=text,
            title=title,
            point=label.lower(),
        )
        for label, text in _labelled_rows(div)
    )
    return chunks


def parse_article(div) -> list[Chunk]:
    """Parse one ``eli-subdivision`` article element into chunks."""
    number = _article_number(div)
    if number is None:
        return []

    _strip_footnotes(div)
    title = _title(div)

    if div.xpath("./div[@id]") and any(
        _PARAGRAPH_ID.match(d.get("id") or "") for d in div.xpath("./div[@id]")
    ):
        return _parse_paragraphs(div, number, title)
    if (title or "").lower().startswith("definitions"):
        return _parse_definitions(div, number, title)
    return _parse_body(div, number, title)


def parse_document(source: str | Path) -> list[Chunk]:
    """Parse a saved EUR-Lex HTML document into citable chunks.

    ``source`` may be a path or the HTML itself.
    """
    # lxml refuses str input carrying an encoding declaration, and these
    # documents open with <?xml version="1.0" encoding="UTF-8"?>. Bytes let it
    # honour the declaration instead of guessing.
    data = Path(source).read_bytes() if _is_path(source) else source.encode("utf-8")
    root = lxml_html.fromstring(data)
    articles = root.xpath('//div[contains(@class, "eli-subdivision")][starts-with(@id, "art_")]')
    return [chunk for article in articles for chunk in parse_article(article)]


def _is_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    return "<" not in source[:200]
