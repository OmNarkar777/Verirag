"""
evaluator/metrics.py — RAGAS metric definitions, thresholds, and interpretations.

Centralizes metric config: thresholds, classifications, human-readable explanations.
RAGAS metric objects are guarded behind a lazy import — RAGAS uses asyncio.get_event_loop()
in thread pool workers which raises RuntimeError on Python 3.12.
Evaluation scoring uses direct Groq LLM calls (see routers/eval.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricThreshold:
    """Pass/fail thresholds for each RAGAS metric."""
    metric_name: str
    warning_threshold: float
    failure_threshold: float
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


def classify_score(metric_name: str, score: float) -> str:
    """Returns 'pass' | 'warning' | 'fail' for a given metric score."""
    threshold = METRIC_THRESHOLDS.get(metric_name)
    if not threshold:
        return "unknown"
    if score >= threshold.warning_threshold:
        return "pass"
    elif score >= threshold.failure_threshold:
        return "warning"
    return "fail"


def score_summary(scores: dict[str, float | None]) -> dict[str, dict]:
    """
    Returns enriched score dict with status classifications.

    Input:  {"faithfulness": 0.85, "answer_relevancy": 0.72, ...}
    Output: {"faithfulness": {"score": 0.85, "status": "pass", "description": "..."}, ...}
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
