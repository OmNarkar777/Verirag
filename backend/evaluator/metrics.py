"""
evaluator/metrics.py — RAGAS metric definitions, thresholds, and interpretations.

WHY THIS FILE EXISTS:
Instead of scattering metric config across the codebase, we centralize:
1. Which RAGAS metrics to run (some are expensive — context_recall calls LLM twice)
2. Thresholds for pass/fail classification (used in dashboards + alerting)
3. Human-readable explanations (for API responses and reports)

RAGAS METRIC PRIMER (what interviewers actually want you to understand):

FAITHFULNESS (0-1):
  "Does the answer contain ONLY information present in the retrieved context?"
  Calculated by: LLM decomposes answer into atomic claims, then judges each
  claim as supported/unsupported by context. Score = supported_claims / total_claims.
  Common failure: LLM uses parametric knowledge instead of retrieved context.
  Fix: stronger system prompt, better context coverage.

ANSWER RELEVANCY (0-1):
  "Does the answer actually address the question asked?"
  Calculated by: LLM generates N reverse questions from the answer, then
  measures cosine similarity between original question and reverse questions.
  Common failure: answer is factually correct but off-topic.
  Fix: better retrieval (getting relevant chunks), prompt improvements.

CONTEXT PRECISION (0-1):
  "Are the USEFUL chunks ranked HIGHER in the retrieved list?"
  Calculated by: for each retrieved chunk, LLM judges if it was needed for
  the answer. Then measures ranking quality (useful chunks should be rank 1,2,3...).
  Common failure: retriever returns many irrelevant chunks mixed with good ones.
  Fix: better embedding model, re-ranking, MMR.

CONTEXT RECALL (0-1):
  "Does the retrieved context COVER everything in the ground truth?"
  Calculated by: LLM decomposes ground truth into sentences, judges each as
  supported/unsupported by retrieved context. Score = supported / total.
  Common failure: retrieved chunks miss key information from the corpus.
  Fix: larger top_k, better chunking strategy, hybrid search.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


@dataclass
class MetricThreshold:
    """
    Pass/fail thresholds for each RAGAS metric.
    
    These are domain-dependent — a customer service bot needs higher faithfulness
    (never hallucinate policy info) than a creative writing assistant.
    Values here are reasonable defaults for a general-purpose RAG system.
    """
    metric_name: str
    warning_threshold: float   # Below this: flag as warning
    failure_threshold: float   # Below this: flag as failure
    description: str


METRIC_THRESHOLDS: dict[str, MetricThreshold] = {
    "faithfulness": MetricThreshold(
        metric_name="faithfulness",
        warning_threshold=0.80,
        failure_threshold=0.60,
        description=(
            "Measures if the answer is grounded in retrieved context. "
            "< 0.60 indicates significant hallucination risk."
        ),
    ),
    "answer_relevancy": MetricThreshold(
        metric_name="answer_relevancy",
        warning_threshold=0.75,
        failure_threshold=0.50,
        description=(
            "Measures if the answer addresses the question. "
            "< 0.50 indicates the pipeline is returning off-topic answers."
        ),
    ),
    "context_precision": MetricThreshold(
        metric_name="context_precision",
        warning_threshold=0.70,
        failure_threshold=0.50,
        description=(
            "Measures retrieval ranking quality. "
            "< 0.50 indicates relevant chunks are not being prioritized."
        ),
    ),
    "context_recall": MetricThreshold(
        metric_name="context_recall",
        warning_threshold=0.75,
        failure_threshold=0.55,
        description=(
            "Measures retrieval coverage vs ground truth. "
            "< 0.55 indicates the corpus is missing key information."
        ),
    ),
}


def get_ragas_metrics() -> list:
    """
    Returns RAGAS metric instances configured for our eval pipeline.
    
    NOTE: RAGAS metrics are stateful objects that hold the LLM reference.
    We set the LLM on them in ragas_runner.py (after settings are loaded),
    not here — this function just defines WHICH metrics to run.
    """
    return [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]


def classify_score(metric_name: str, score: float) -> str:
    """
    Returns "pass" | "warning" | "fail" for a given metric score.
    Used for dashboard color coding and alerting.
    """
    threshold = METRIC_THRESHOLDS.get(metric_name)
    if not threshold:
        return "unknown"

    if score >= threshold.warning_threshold:
        return "pass"
    elif score >= threshold.failure_threshold:
        return "warning"
    else:
        return "fail"


def score_summary(scores: dict[str, float | None]) -> dict[str, dict]:
    """
    Returns enriched score dict with classifications.
    
    Input: {"faithfulness": 0.85, "answer_relevancy": 0.72, ...}
    Output: {
        "faithfulness": {"score": 0.85, "status": "pass", "description": "..."},
        ...
    }
    """
    summary = {}
    for metric_name, score in scores.items():
        threshold = METRIC_THRESHOLDS.get(metric_name)
        summary[metric_name] = {
            "score": score,
            "status": classify_score(metric_name, score) if score is not None else "unknown",
            "description": threshold.description if threshold else "",
        }
    return summary
