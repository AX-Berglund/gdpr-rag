"""Tests for the EUR-Lex Official Journal HTML parser.

The fixture mirrors the real markup exactly -- same classes, same id scheme,
same two-cell tables -- but is small enough to read. The real document is not
vendored, so a `scripts/validate_corpus.py` run is what checks this against all
99 articles.
"""

import pytest

from gdpr_rag.ingest.eurlex import parse_document
from gdpr_rag.ingest.models import ChunkKind

FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<html><body>
  <div class="eli-subdivision" id="art_4">
    <p class="oj-ti-art">Article 4</p>
    <div class="eli-title" id="art_4.tit_1"><p class="oj-sti-art">Definitions</p></div>
    <p class="oj-normal">For the purposes of this Regulation:</p>
    <table><tbody><tr>
      <td><p class="oj-normal">(1)</p></td>
      <td><p class="oj-normal">&#8216;personal data&#8217; means any information
         relating to a natural person;</p></td>
    </tr></tbody></table>
    <table><tbody><tr>
      <td><p class="oj-normal">(7)</p></td>
      <td><p class="oj-normal">&#8216;controller&#8217; means the person which
         determines the purposes;</p></td>
    </tr></tbody></table>
  </div>

  <div class="eli-subdivision" id="art_17">
    <p class="oj-ti-art">Article 17</p>
    <div class="eli-title" id="art_17.tit_1">
      <p class="oj-sti-art">Right to erasure (&#8216;right to be forgotten&#8217;)</p>
    </div>
    <div id="017.001">
      <p class="oj-normal">1.&#160;&#160; The data subject shall have the right to obtain erasure
         <span class="oj-super oj-note-tag">12</span> without undue delay:</p>
      <table><tbody><tr>
        <td><p class="oj-normal">(a)</p></td>
        <td><p class="oj-normal">the personal data are no longer necessary;</p></td>
      </tr></tbody></table>
      <table><tbody><tr>
        <td><p class="oj-normal">(b)</p></td>
        <td><p class="oj-normal">the data subject withdraws consent;</p></td>
      </tr></tbody></table>
    </div>
    <div id="017.002">
      <p class="oj-normal">2.&#160;&#160; Where the controller has made the data public.</p>
    </div>
  </div>

  <div class="eli-subdivision" id="art_50">
    <p class="oj-ti-art">Article 50</p>
    <div class="eli-title" id="art_50.tit_1">
      <p class="oj-sti-art">International cooperation</p>
    </div>
    <p class="oj-normal">The Commission shall take appropriate steps to develop cooperation.</p>
  </div>

  <div class="eli-subdivision" id="rct_1">
    <p class="oj-normal">A recital, which is not an article and must be skipped.</p>
  </div>
</body></html>
"""


@pytest.fixture(scope="module")
def chunks():
    return parse_document(FIXTURE)


class TestDocumentScope:
    def test_only_articles_are_parsed(self, chunks):
        assert {c.article for c in chunks} == {4, 17, 50}

    def test_recitals_are_skipped(self, chunks):
        assert not any("recital" in c.text.lower() for c in chunks)


class TestParagraphArticles:
    def test_paragraphs_and_points_are_separate_chunks(self, chunks):
        citations = [c.citation for c in chunks if c.article == 17]
        assert citations == [
            "Article 17(1)",
            "Article 17(1)(a)",
            "Article 17(1)(b)",
            "Article 17(2)",
        ]

    def test_paragraph_number_prefix_is_stripped(self, chunks):
        lead = next(c for c in chunks if c.citation == "Article 17(1)")
        assert lead.text.startswith("The data subject")

    def test_footnote_markers_are_removed(self, chunks):
        lead = next(c for c in chunks if c.citation == "Article 17(1)")
        assert "12" not in lead.text
        assert "erasure without undue delay" in lead.text

    def test_title_is_attached(self, chunks):
        assert all(
            c.title == "Right to erasure (‘right to be forgotten’)"
            for c in chunks
            if c.article == 17
        )

    def test_point_text_excludes_its_own_label(self, chunks):
        point = next(c for c in chunks if c.citation == "Article 17(1)(a)")
        assert point.text == "the personal data are no longer necessary;"


class TestDefinitions:
    def test_definitions_keep_their_numbering(self, chunks):
        assert [c.citation for c in chunks if c.article == 4] == [
            "Article 4(1)",
            "Article 4(7)",
        ]

    def test_kind_is_definition(self, chunks):
        assert all(c.kind is ChunkKind.DEFINITION for c in chunks if c.article == 4)

    def test_lead_in_is_not_emitted_as_a_definition(self, chunks):
        assert not any(c.text.startswith("For the purposes") for c in chunks)


class TestBodyArticles:
    def test_prose_article_is_a_single_body_chunk(self, chunks):
        body = [c for c in chunks if c.article == 50]
        assert len(body) == 1
        assert body[0].kind is ChunkKind.BODY
        assert body[0].citation == "Article 50"


class TestTextQuality:
    def test_no_newlines_survive(self, chunks):
        assert not any("\n" in c.text for c in chunks)

    def test_no_doubled_spaces_survive(self, chunks):
        assert not any("  " in c.text for c in chunks)

    def test_non_breaking_spaces_are_normalised(self, chunks):
        assert not any("\xa0" in c.text for c in chunks)

    def test_no_chunk_is_empty(self, chunks):
        assert all(c.text.strip() for c in chunks)


class TestInputForms:
    def test_accepts_a_path(self, tmp_path):
        path = tmp_path / "doc.html"
        path.write_text(FIXTURE, encoding="utf-8")
        assert parse_document(path)

    def test_accepts_markup_with_an_xml_declaration(self):
        # lxml rejects str carrying an encoding declaration; this must not leak out.
        assert parse_document(FIXTURE)
