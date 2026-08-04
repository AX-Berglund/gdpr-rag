"""Tests for the labelled evaluation set and its loader.

These also act as a lint pass over questions.yaml itself: a malformed label
fails the build rather than quietly skewing a published metric.
"""

import pytest
from pydantic import ValidationError

from gdpr_rag.evaluation import Question, load_questions


@pytest.fixture(scope="module")
def questions():
    return load_questions()


class TestShippedSet:
    def test_loads(self, questions):
        assert len(questions) >= 60

    def test_ids_are_unique(self, questions):
        assert len({q.id for q in questions}) == len(questions)

    def test_every_answerable_question_has_labels(self, questions):
        assert all(q.articles for q in questions if not q.unanswerable)

    def test_refusal_set_is_present(self, questions):
        assert sum(q.unanswerable for q in questions) >= 4

    def test_all_three_difficulties_are_represented(self, questions):
        assert {q.difficulty for q in questions} == {"easy", "medium", "hard"}

    def test_labels_are_plausible_article_numbers(self, questions):
        # The GDPR has 99 articles.
        assert all(1 <= n <= 99 for q in questions for n in q.article_numbers)


class TestValidation:
    def test_paragraph_level_labels_are_rejected(self):
        with pytest.raises(ValidationError, match="article-level"):
            Question(id="x", question="q", articles=["Article 17(1)"])

    def test_answerable_question_without_labels_is_rejected(self):
        with pytest.raises(ValidationError, match="needs at least one article label"):
            Question(id="x", question="q", articles=[])

    def test_unanswerable_question_with_labels_is_rejected(self):
        with pytest.raises(ValidationError, match="cannot have article labels"):
            Question(id="x", question="q", articles=["Article 17"], unanswerable=True)

    def test_unknown_difficulty_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown difficulty"):
            Question(id="x", question="q", articles=["Article 5"], difficulty="trivial")

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "absent.yaml")

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "q.yaml"
        path.write_text(
            "- {id: a, question: x, articles: [Article 5]}\n"
            "- {id: a, question: y, articles: [Article 6]}\n"
        )
        with pytest.raises(ValueError, match="duplicate question ids"):
            load_questions(path)


class TestQuestion:
    def test_article_numbers_are_parsed(self):
        q = Question(id="x", question="q", articles=["Article 17", "Article 4"])
        assert q.article_numbers == {17, 4}


class TestPhrasing:
    def test_every_question_has_a_colloquial_variant(self, questions):
        assert all(q.colloquial for q in questions)

    def test_colloquial_variants_are_actually_different(self, questions):
        assert all(q.colloquial != q.question for q in questions)

    def test_variants_look_like_something_a_user_would_type(self, questions):
        # Lowercase and unpunctuated is the point; a polished sentence here
        # would measure nothing.
        assert all(not q.colloquial.endswith("?") for q in questions)
        assert all(q.colloquial[0].islower() for q in questions)

    def test_formal_phrasing_is_the_default(self, questions):
        assert questions[0].phrased() == questions[0].question

    def test_colloquial_phrasing_is_selectable(self, questions):
        assert questions[0].phrased("colloquial") == questions[0].colloquial

    def test_missing_variant_falls_back_rather_than_dropping_the_question(self):
        # Silently dropping one would shift the mean it contributes to.
        bare = Question(id="x", question="Formal?", articles=["Article 5"])
        assert bare.phrased("colloquial") == "Formal?"

    def test_unknown_style_is_rejected(self, questions):
        with pytest.raises(ValueError, match="unknown phrasing style"):
            questions[0].phrased("shouty")
