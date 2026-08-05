"""The labelled evaluation set and the metrics computed over it."""

from gdpr_rag.evaluation.cases import DEFAULT_CASES, Case, CaseReport, evaluate_cases, load_cases
from gdpr_rag.evaluation.dataset import DEFAULT_QUESTIONS, Question, load_questions
from gdpr_rag.evaluation.metrics import hit_rate, ndcg, reciprocal_rank, to_article_ranking
from gdpr_rag.evaluation.report import QuestionResult, RetrievalReport, evaluate_retrieval
from gdpr_rag.evaluation.scenarios import DEFAULT_SCENARIOS, Scenario, coverage, load_scenarios

__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_QUESTIONS",
    "DEFAULT_SCENARIOS",
    "Case",
    "CaseReport",
    "Question",
    "QuestionResult",
    "RetrievalReport",
    "Scenario",
    "coverage",
    "evaluate_cases",
    "evaluate_retrieval",
    "hit_rate",
    "load_cases",
    "load_questions",
    "load_scenarios",
    "ndcg",
    "reciprocal_rank",
    "to_article_ranking",
]
