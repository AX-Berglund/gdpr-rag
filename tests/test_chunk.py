"""Tests for structure-aware chunking.

The fixtures are abridged but structurally faithful excerpts of the GDPR --
enough shape to exercise each parsing path without vendoring the whole text.
"""

import pytest

from gdpr_rag.ingest.chunk import chunk_document, classify, parse_article, split_articles
from gdpr_rag.ingest.models import ChunkKind

ARTICLE_17 = """Article 17
Right to erasure ('right to be forgotten')
1. The data subject shall have the right to obtain from the controller the erasure of personal
data concerning him or her without undue delay where one of the following grounds applies:
(a) the personal data are no longer necessary in relation to the purposes for which they were
collected;
(b) the data subject withdraws consent on which the processing is based;
2. Where the controller has made the personal data public, the controller shall take reasonable
steps to inform other controllers.
"""

ARTICLE_4 = """Article 4
Definitions
For the purposes of this Regulation:
(1) 'personal data' means any information relating to an identified or identifiable natural
person;
(2) 'processing' means any operation which is performed on personal data;
(7) 'controller' means the natural or legal person which determines the purposes of the
processing;
"""

ARTICLE_1 = """Article 1
Subject-matter and objectives
This Regulation lays down rules relating to the protection of natural persons with regard to
the processing of personal data.
"""


class TestClassify:
    def test_numbered_paragraphs_are_paragraph_articles(self):
        assert classify(ARTICLE_17) is ChunkKind.PARAGRAPH

    def test_definitions_article_is_detected(self):
        assert classify(ARTICLE_4) is ChunkKind.DEFINITION

    def test_prose_article_is_body(self):
        assert classify(ARTICLE_1) is ChunkKind.BODY


class TestParagraphArticles:
    def test_lead_in_and_points_become_separate_chunks(self):
        chunks = parse_article(ARTICLE_17)
        citations = [c.citation for c in chunks]
        assert "Article 17(1)" in citations
        assert "Article 17(1)(a)" in citations
        assert "Article 17(1)(b)" in citations
        assert "Article 17(2)" in citations

    def test_points_carry_their_own_text_only(self):
        chunks = {c.citation: c.text for c in parse_article(ARTICLE_17)}
        assert chunks["Article 17(1)(a)"].startswith("the personal data are no longer necessary")
        # The lead-in must not swallow the points that follow it.
        assert "no longer necessary" not in chunks["Article 17(1)"]

    def test_title_is_attached_to_every_chunk(self):
        chunks = parse_article(ARTICLE_17)
        assert all(c.title == "Right to erasure ('right to be forgotten')" for c in chunks)

    def test_paragraph_without_points_is_a_single_chunk(self):
        chunks = [c for c in parse_article(ARTICLE_17) if c.paragraph == "2"]
        assert len(chunks) == 1
        assert chunks[0].point is None


class TestDefinitionArticles:
    def test_each_definition_is_its_own_chunk(self):
        chunks = parse_article(ARTICLE_4)
        assert [c.definition for c in chunks] == ["1", "2", "7"]

    def test_definition_numbering_is_preserved_not_reindexed(self):
        # Definition (7) must cite as (7), not as the third item parsed.
        controller = next(c for c in parse_article(ARTICLE_4) if "controller" in c.text)
        assert controller.citation == "Article 4(7)"

    def test_lead_in_is_not_emitted_as_a_definition(self):
        texts = [c.text for c in parse_article(ARTICLE_4)]
        assert not any(t.startswith("For the purposes") for t in texts)


class TestBodyArticles:
    def test_prose_article_yields_one_chunk(self):
        chunks = parse_article(ARTICLE_1)
        assert len(chunks) == 1
        assert chunks[0].citation == "Article 1"
        assert chunks[0].kind is ChunkKind.BODY

    def test_line_wrapping_is_joined(self):
        text = parse_article(ARTICLE_1)[0].text
        assert "\n" not in text
        assert "natural persons with regard to the processing" in text


class TestSplitArticles:
    def test_articles_are_separated(self):
        document = f"{ARTICLE_1}\n{ARTICLE_4}\n{ARTICLE_17}"
        assert [n for n, _ in split_articles(document)] == [1, 4, 17]

    def test_preamble_before_first_article_is_dropped(self):
        document = f"Having regard to the Treaty on the Functioning of the Union,\n\n{ARTICLE_1}"
        articles = list(split_articles(document))
        assert len(articles) == 1
        assert "Having regard" not in articles[0][1]

    def test_document_chunking_covers_all_articles(self):
        document = f"{ARTICLE_1}\n{ARTICLE_4}\n{ARTICLE_17}"
        assert {c.article for c in chunk_document(document)} == {1, 4, 17}


class TestChunkInvariants:
    @pytest.mark.parametrize("article", [ARTICLE_1, ARTICLE_4, ARTICLE_17])
    def test_no_chunk_is_empty(self, article):
        assert all(c.text.strip() for c in parse_article(article))

    @pytest.mark.parametrize("article", [ARTICLE_1, ARTICLE_4, ARTICLE_17])
    def test_every_chunk_is_citable(self, article):
        assert all(c.citation.startswith("Article ") for c in parse_article(article))

    def test_missing_header_is_an_error(self):
        with pytest.raises(ValueError, match="No article header"):
            parse_article("1. Some text with no article header.")
