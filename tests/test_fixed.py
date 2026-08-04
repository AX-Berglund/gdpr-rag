"""Tests for the fixed-size baseline chunker."""

import pytest

from gdpr_rag.ingest.fixed import fixed_size_chunks
from gdpr_rag.ingest.models import Chunk, ChunkKind


def chunk(article: int, text: str) -> Chunk:
    return Chunk(article=article, kind=ChunkKind.PARAGRAPH, text=text)


SOURCE = [
    chunk(17, "A" * 500),
    chunk(18, "B" * 500),
    chunk(19, "C" * 500),
]


class TestWindowing:
    def test_windows_cover_the_whole_document(self):
        joined = "".join(c.text for c in fixed_size_chunks(SOURCE, size=400, overlap=0))
        assert joined.count("A") == 500
        assert joined.count("C") == 500

    def test_overlap_repeats_content(self):
        no_overlap = fixed_size_chunks(SOURCE, size=400, overlap=0)
        with_overlap = fixed_size_chunks(SOURCE, size=400, overlap=200)
        assert len(with_overlap) > len(no_overlap)

    def test_window_size_is_respected(self):
        assert all(len(c.text) <= 400 for c in fixed_size_chunks(SOURCE, size=400, overlap=50))

    def test_single_window_when_size_exceeds_document(self):
        assert len(fixed_size_chunks(SOURCE, size=10_000, overlap=0)) == 1


class TestAttribution:
    def test_windows_are_attributed_to_an_article(self):
        assert all(c.article in {17, 18, 19} for c in fixed_size_chunks(SOURCE, size=200))

    def test_early_windows_belong_to_the_first_article(self):
        assert fixed_size_chunks(SOURCE, size=100, overlap=0)[0].article == 17

    def test_late_windows_belong_to_the_last_article(self):
        assert fixed_size_chunks(SOURCE, size=100, overlap=0)[-1].article == 19


class TestCitationLimits:
    def test_windows_carry_no_structural_address(self):
        # The baseline cannot cite a paragraph or point — this is the point.
        chunks = fixed_size_chunks(SOURCE, size=300)
        assert all(c.paragraph is None and c.point is None for c in chunks)

    def test_citation_is_article_level_only(self):
        assert all(c.citation.count("(") == 0 for c in fixed_size_chunks(SOURCE, size=300))


class TestValidation:
    def test_empty_source_gives_no_windows(self):
        assert fixed_size_chunks([]) == []

    def test_blank_chunks_are_skipped(self):
        assert fixed_size_chunks([chunk(1, "   ")]) == []

    @pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0)])
    def test_invalid_size_is_rejected(self, size, overlap):
        with pytest.raises(ValueError, match="size must be positive"):
            fixed_size_chunks(SOURCE, size=size, overlap=overlap)

    @pytest.mark.parametrize("overlap", [-1, 400, 500])
    def test_invalid_overlap_is_rejected(self, overlap):
        with pytest.raises(ValueError, match="overlap must be"):
            fixed_size_chunks(SOURCE, size=400, overlap=overlap)
