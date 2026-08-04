"""Fixed-size chunking: the baseline that structured chunking has to beat.

This is what most RAG pipelines do — slide a window of N characters over the
document and embed each window. It is included so the project's central claim
is measured rather than asserted.

The baseline is built to be *fair*, not to lose. It reads the same clean text
the structured parser produces, so no part of any difference comes from
extraction quality. Windows are allowed to cross article boundaries, as a real
naive pipeline's would, and each is attributed to the article containing its
midpoint.

What it cannot do is cite precisely. A window has no structural address, so the
best it can offer is "Article 17" — never "Article 17(1)(a)". That limitation is
part of the result, not a flaw in the comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

from gdpr_rag.ingest.models import Chunk, ChunkKind


def fixed_size_chunks(
    source: Sequence[Chunk],
    size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Re-split the corpus into overlapping fixed-size windows.

    ``source`` supplies both the text and the article each span belongs to, so
    the baseline sees exactly the same characters as the structured chunker.
    """
    if size < 1:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not source:
        return []

    # Rebuild the document, remembering which article owns each character span.
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (start, end, article)
    cursor = 0
    for chunk in source:
        text = chunk.text.strip()
        if not text:
            continue
        parts.append(text)
        spans.append((cursor, cursor + len(text), chunk.article))
        cursor += len(text) + 1  # the joining space
    document = " ".join(parts)

    step = size - overlap
    windows = []
    for start in range(0, len(document), step):
        text = document[start : start + size].strip()
        if not text:
            continue
        windows.append(
            Chunk(
                article=_article_at(spans, start + min(size, len(document) - start) // 2),
                kind=ChunkKind.BODY,
                text=text,
            )
        )
        if start + size >= len(document):
            break
    return windows


def _article_at(spans: list[tuple[int, int, int]], position: int) -> int:
    """The article owning ``position``, or the nearest preceding one."""
    best = spans[0][2]
    for start, end, article in spans:
        if start <= position < end:
            return article
        if start <= position:
            best = article
    return best
