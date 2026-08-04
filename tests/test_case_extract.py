"""Tests for reading evaluation labels out of Court of Justice judgments."""

import pytest

from gdpr_rag.cases.extract import (
    NotAGdprCase,
    extract,
    is_gdpr_case,
    to_text,
)

GDPR_JUDGMENT = """
<p>JUDGMENT OF THE COURT (Third Chamber) 4 May 2023</p>
<p>(Reference for a preliminary ruling &#8211; Protection of natural persons with regard to
the processing of personal data &#8211; Regulation (EU) 2016/679 &#8211; Article 82(1) &#8211;
Right to compensation for damage &#8211; Charter of Fundamental Rights &#8211; Article 47)</p>
<p>In Case C-300/21, the referring court decided to stay the proceedings and to refer the
following questions to the Court of Justice for a preliminary ruling:
(1) Does the award of compensation under Article 82 of Regulation 2016/679 require, in
addition to an infringement, that the applicant has suffered harm of some kind?
(2) Are there further requirements under EU law which must be observed when assessing
the amount of compensation payable under Article 82 of that regulation?</p>
<p>Article 82 of Regulation 2016/679 must be interpreted as meaning that a mere
infringement does not confer a right to compensation. Article 82 requires damage.
Article 82 does not permit punitive damages. Article 82 applies here. Article 82 again.
Article 83 concerns fines and is not at issue. Article 4 defines the terms.</p>
"""

SUCCESSION_JUDGMENT = """
<p>JUDGMENT OF THE COURT (First Chamber) 7 April 2022</p>
<p>(Reference for a preliminary ruling &#8211; Judicial cooperation in civil matters &#8211;
Regulation (EU) No 650/2012 &#8211; Article 10 &#8211; Subsidiary jurisdiction in matters
of succession)</p>
<p>Article 10 of Regulation No 650/2012 must be interpreted as meaning that Article 10
applies. Article 10 again. Article 10 once more. Article 10 finally.</p>
"""


class TestScoping:
    def test_a_gdpr_judgment_is_recognised(self):
        assert is_gdpr_case(to_text(GDPR_JUDGMENT))

    def test_a_judgment_about_another_instrument_is_not(self):
        assert not is_gdpr_case(to_text(SUCCESSION_JUDGMENT))

    def test_extracting_a_non_gdpr_judgment_raises(self):
        # An earlier version happily labelled a succession case "Article 10".
        with pytest.raises(NotAGdprCase, match="2016/679"):
            extract("62020CJ0645", SUCCESSION_JUDGMENT)


class TestLabels:
    @pytest.fixture(scope="class")
    def result(self):
        return extract("62021CJ0300", GDPR_JUDGMENT)

    def test_the_headnote_supplies_the_labels(self, result):
        assert result.articles == [82]
        assert result.label_source == "headnote"

    def test_articles_of_other_instruments_are_excluded(self, result):
        # Article 47 of the Charter appears in the same headnote.
        assert 47 not in result.articles

    def test_mention_counts_are_recorded_for_auditing(self, result):
        assert result.mentions[82] > result.mentions.get(83, 0)

    def test_procedural_articles_are_dropped(self):
        from gdpr_rag.cases.extract import PROCEDURAL_ARTICLES

        assert 60 in PROCEDURAL_ARTICLES
        assert 17 not in PROCEDURAL_ARTICLES


class TestReferredQuestions:
    def test_the_courts_own_questions_are_captured(self):
        # Using the referring court's wording keeps the query side external too.
        questions = extract("62021CJ0300", GDPR_JUDGMENT).questions
        assert len(questions) == 2
        assert questions[0].endswith("?")
        assert "compensation" in questions[0]

    def test_absent_questions_give_an_empty_list(self):
        minimal = "<p>Regulation (EU) 2016/679 &#8211; Article 5 was considered.</p>"
        assert extract("x", minimal).questions == []


class TestTextFlattening:
    def test_tags_are_stripped_and_entities_decoded(self):
        assert to_text("<p>a&#8211;b</p>") == "a–b"

    def test_whitespace_is_collapsed(self):
        assert to_text("<p>a\n\n   b</p>") == "a b"
