"""Tests for multi-query retrieval: decomposition and rank fusion."""

import pytest

from gdpr_rag.ingest.models import Chunk, ChunkKind
from gdpr_rag.multiquery import Decomposer, multi_query_retrieve, reciprocal_rank_fusion
from gdpr_rag.store.sqlite_store import SearchResult


class ScriptedModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def result(article: int, score: float = 0.5, paragraph=None):
    return SearchResult(
        chunk=Chunk(
            article=article,
            kind=ChunkKind.PARAGRAPH if paragraph else ChunkKind.BODY,
            text=f"text of article {article}",
            paragraph=paragraph,
        ),
        score=score,
    )


class TestDecomposer:
    def test_sub_questions_are_returned_with_the_original_first(self):
        model = ScriptedModel("Is the processing lawful?\nWas the person informed?")
        subs = Decomposer(model).decompose("Can I email them again?")
        assert subs == [
            "Can I email them again?",
            "Is the processing lawful?",
            "Was the person informed?",
        ]

    def test_the_original_is_kept_because_decomposition_can_lose_the_thread(self):
        # Every part answerable and none of them the question asked is a real
        # outcome, so the question itself always stays in the query set.
        subs = Decomposer(ScriptedModel("A?\nB?")).decompose("original?")
        assert subs[0] == "original?"

    @pytest.mark.parametrize(
        "line",
        [
            "1. Is it lawful?",
            "- Is it lawful?",
            "• Is it lawful?",
            '"Is it lawful?"',
            "  Is it lawful?  ",
        ],
    )
    def test_list_decoration_is_stripped(self, line):
        subs = Decomposer(ScriptedModel(line)).decompose("q")
        assert subs[1] == "Is it lawful?"

    def test_blank_lines_are_dropped(self):
        subs = Decomposer(ScriptedModel("A?\n\n\nB?\n")).decompose("q")
        assert subs == ["q", "A?", "B?"]

    def test_a_model_repeating_the_question_does_not_duplicate_it(self):
        subs = Decomposer(ScriptedModel("q\nSomething else?")).decompose("q")
        assert subs == ["q", "Something else?"]

    def test_the_limit_is_respected_and_reaches_the_prompt(self):
        model = ScriptedModel("A?\nB?\nC?\nD?\nE?")
        subs = Decomposer(model, limit=2).decompose("q")
        assert subs == ["q", "A?", "B?"]
        assert "at most 2" in model.last_prompt

    def test_a_model_returning_nothing_leaves_just_the_question(self):
        assert Decomposer(ScriptedModel("")).decompose("q") == ["q"]


class TestReciprocalRankFusion:
    def test_a_single_ranking_is_returned_in_order(self):
        ranking = [result(17), result(15), result(6)]
        fused = reciprocal_rank_fusion([ranking])
        assert [r.chunk.article for r in fused] == [17, 15, 6]

    def test_agreement_across_rankings_outranks_one_strong_showing(self):
        """The property the whole method rests on.

        Article 15 is second in both rankings; article 17 is first in one and
        absent from the other. Consistent support wins, because agreement
        across phrasings is evidence and one query's enthusiasm is not.
        """
        fused = reciprocal_rank_fusion([[result(17), result(15)], [result(6), result(15)]])
        assert fused[0].chunk.article == 15

    def test_scores_are_not_compared_across_rankings(self):
        # Article 6 has a far higher similarity, but it is last in its ranking
        # in both. Rank is what fuses; the raw score is not comparable.
        fused = reciprocal_rank_fusion(
            [[result(17, score=0.2), result(6, score=0.99)], [result(17, score=0.2)]]
        )
        assert fused[0].chunk.article == 17

    def test_the_best_scoring_instance_of_a_chunk_survives(self):
        fused = reciprocal_rank_fusion([[result(17, score=0.3)], [result(17, score=0.8)]])
        assert fused[0].score == 0.8

    def test_chunks_are_merged_by_citation_not_by_object(self):
        fused = reciprocal_rank_fusion([[result(17)], [result(17)]])
        assert len(fused) == 1

    def test_distinct_paragraphs_of_one_article_stay_distinct(self):
        fused = reciprocal_rank_fusion([[result(17, paragraph="1"), result(17, paragraph="2")]])
        assert len(fused) == 2

    def test_empty_rankings_fuse_to_nothing(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_a_nonpositive_constant_is_rejected(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            reciprocal_rank_fusion([[result(17)]], k=0)


class FakeRetriever:
    """Returns a canned ranking per query, and records what it was asked."""

    def __init__(self, rankings: dict[str, list[SearchResult]]) -> None:
        self._rankings = rankings
        self.queries: list[str] = []

    def retrieve(self, question, k=5, trace=None):
        self.queries.append(question)
        return self._rankings.get(question, [])[:k]


class TestMultiQueryRetrieve:
    def test_every_query_is_issued(self):
        retriever = FakeRetriever({"a": [result(17)], "b": [result(15)]})
        multi_query_retrieve(retriever, ["a", "b"])
        assert retriever.queries == ["a", "b"]

    def test_the_budget_is_matched_to_a_single_query(self):
        """Two queries must not be allowed to return twice the evidence.

        Otherwise any measurement at fixed k flatters the method: it would be
        scoring more chunks, not better ones.
        """
        retriever = FakeRetriever(
            {
                "a": [result(1), result(2), result(3)],
                "b": [result(4), result(5), result(6)],
            }
        )
        assert len(multi_query_retrieve(retriever, ["a", "b"], k=3)) == 3

    def test_results_from_different_queries_are_combined(self):
        retriever = FakeRetriever({"a": [result(17)], "b": [result(15)]})
        fused = multi_query_retrieve(retriever, ["a", "b"], k=5)
        assert {r.chunk.article for r in fused} == {17, 15}

    def test_no_queries_is_an_error_rather_than_an_empty_answer(self):
        with pytest.raises(ValueError, match="at least one query"):
            multi_query_retrieve(FakeRetriever({}), [])

    def test_fusion_is_traced(self):
        from gdpr_rag.trace import Trace

        trace = Trace("q")
        retriever = FakeRetriever({"a": [result(17)]})
        multi_query_retrieve(retriever, ["a"], trace=trace)
        span = trace.find("fuse")[0]
        assert span.inputs["queries"] == ["a"]
        assert span.outputs["citations"] == ["Article 17"]
