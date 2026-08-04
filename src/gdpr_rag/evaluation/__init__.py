"""The labelled evaluation set and the metrics computed over it."""

from gdpr_rag.evaluation.dataset import DEFAULT_QUESTIONS, Question, load_questions
from gdpr_rag.evaluation.metrics import hit_rate, ndcg, reciprocal_rank, to_article_ranking
from gdpr_rag.evaluation.report import QuestionResult, RetrievalReport, evaluate_retrieval

__all__ = [
    "DEFAULT_QUESTIONS",
    "Question",
    "QuestionResult",
    "RetrievalReport",
    "evaluate_retrieval",
    "hit_rate",
    "load_questions",
    "ndcg",
    "reciprocal_rank",
    "to_article_ranking",
]
