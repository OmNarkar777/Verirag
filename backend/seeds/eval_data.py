"""
seeds/eval_data.py — Seed 30+ realistic evaluation runs into PostgreSQL.

Creates a realistic history spanning 6 weeks:
- Multiple pipeline configurations (model × strategy × chunk size)
- Score progression showing continuous improvement
- Occasional regression runs with regression_details populated
- 10 eval cases per run with individual scores

All scores are pre-computed (no RAGAS API calls needed). This is standard
practice for populating demo databases with representative historical data.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone

random.seed(42)  # deterministic seed data for consistent results

from loguru import logger
from sqlalchemy import select, func

PIPELINES = [
    ("llama-3.3-70b / mmr / chunk-512", "llama3-groq-70b"),
    ("llama-3.3-70b / similarity / chunk-512", "llama3-groq-70b"),
    ("llama-3.3-70b / mmr / chunk-256", "llama3-groq-70b"),
    ("mixtral-8x7b / mmr / chunk-512", "mixtral-8x7b"),
    ("gemma-7b / similarity / chunk-1024", "gemma-7b-it"),
]

QUESTIONS = [
    "What is the core innovation of the Transformer architecture?",
    "How does RAGAS measure faithfulness in RAG pipelines?",
    "Why does MMR retrieval improve context precision scores?",
    "What are the tradeoffs between chunk size and retrieval quality?",
    "How does HNSW indexing improve vector search performance?",
    "What makes LangChain suitable for building RAG applications?",
    "How does context recall differ from context precision in RAGAS?",
    "What are the key considerations when choosing an embedding model?",
    "How does prompt engineering affect faithfulness scores?",
    "What is the role of the retriever in a RAG evaluation pipeline?",
]

ANSWERS = [
    "The Transformer replaces recurrence with self-attention, enabling parallel processing and capturing long-range dependencies more effectively than RNNs or LSTMs.",
    "RAGAS faithfulness measures whether each claim in the generated answer can be inferred from the retrieved context, using an LLM to verify grounding.",
    "MMR penalizes redundant chunks by maximizing marginal relevance—selecting diverse, relevant contexts rather than near-duplicates, improving RAGAS context precision.",
    "Smaller chunks (128-256 tokens) improve precision but may lose context. Larger chunks (512-1024) provide richer context but can dilute relevance. 512 is a common sweet spot.",
    "HNSW builds a hierarchical graph where each node connects to its nearest neighbors at multiple layers, allowing logarithmic-time approximate nearest-neighbor search.",
    "LangChain provides composable chains, retriever abstractions, document loaders, and LLM integrations that dramatically simplify RAG pipeline construction.",
    "Context recall measures whether all ground-truth information was present in the retrieved context. Context precision measures how much of the retrieved context was actually useful.",
    "Key factors include embedding dimensionality, domain coverage, multilingual support, inference speed, and whether the model was trained on similar domain text.",
    "Clear, specific system prompts that instruct the model to answer only from context improve faithfulness by reducing hallucination from parametric knowledge.",
    "The retriever selects relevant document chunks that become the 'context' for LLM generation. Poor retrieval is the most common source of faithfulness and recall failures.",
]

GROUND_TRUTHS = [
    "The Transformer architecture's key innovation is the self-attention mechanism that processes all tokens in parallel and captures global dependencies without sequential computation.",
    "RAGAS faithfulness score is computed by decomposing the answer into claims and checking what fraction of those claims are supported by the retrieved context.",
    "MMR retrieval selects documents that maximize relevance to the query while minimizing redundancy with already-selected documents, yielding more informative context sets.",
    "Chunk size should balance context richness against retrieval precision; 512 tokens is widely recommended as a starting point with 50-token overlap to avoid boundary effects.",
    "HNSW (Hierarchical Navigable Small World) is a graph-based indexing algorithm that provides sub-linear query time for approximate nearest-neighbor search in high-dimensional spaces.",
    "LangChain is suitable because it provides pre-built abstractions for document loading, text splitting, vector store integration, and LLM chaining that reduce boilerplate code.",
    "Context recall (ground truth coverage in retrieved context) and context precision (fraction of retrieved context that is relevant) measure complementary aspects of retrieval quality.",
    "Embedding model selection should consider: (1) domain match, (2) dimensionality vs. speed tradeoff, (3) multilingual needs, (4) licensing, and (5) benchmark performance on target tasks.",
    "System prompts that explicitly restrict the model to 'answer only from provided context' and instruct it to say 'I don't know' when context is insufficient reduce hallucination.",
    "The retriever's output directly determines what context the LLM sees, making retrieval quality the primary determinant of faithfulness, recall, and precision scores in RAGAS.",
]


def _make_scores(
    base_faithfulness: float,
    base_relevancy: float,
    base_precision: float,
    base_recall: float,
    noise: float = 0.05,
) -> tuple[float, float, float, float]:
    """Generate scores with realistic noise, clamped to [0, 1]."""
    rng = random.random
    f = max(0.0, min(1.0, base_faithfulness + (rng() - 0.5) * noise * 2))
    r = max(0.0, min(1.0, base_relevancy + (rng() - 0.5) * noise * 2))
    p = max(0.0, min(1.0, base_precision + (rng() - 0.5) * noise * 2))
    c = max(0.0, min(1.0, base_recall + (rng() - 0.5) * noise * 2))
    return f, r, p, c


def _run_data(seed_state: dict) -> list[dict]:
    """
    Generate 32 eval run definitions with realistic score trajectories.
    seed_state is mutated to track progression.
    """
    now = datetime.now(timezone.utc)
    runs = []

    configs = [
        # (week_offset, days_into_week, pipeline_idx, base_scores, is_regression)
        # Week 6 ago — baseline (v1.0.x): mediocre
        (6, 0, 0, (0.58, 0.62, 0.52, 0.48), False),
        (6, 2, 1, (0.61, 0.65, 0.55, 0.51), False),
        (6, 4, 2, (0.57, 0.60, 0.50, 0.46), False),
        (6, 6, 3, (0.55, 0.63, 0.48, 0.44), False),

        # Week 5 ago — v1.0.x continued + regression
        (5, 1, 0, (0.63, 0.68, 0.58, 0.53), False),
        (5, 3, 1, (0.47, 0.55, 0.43, 0.39), True),   # regression run
        (5, 5, 2, (0.66, 0.70, 0.60, 0.56), False),
        (5, 6, 4, (0.60, 0.64, 0.54, 0.50), False),

        # Week 4 ago — v1.1.x: noticeable improvement
        (4, 0, 0, (0.72, 0.76, 0.68, 0.63), False),
        (4, 2, 1, (0.75, 0.78, 0.70, 0.65), False),
        (4, 3, 0, (0.74, 0.77, 0.69, 0.64), False),
        (4, 5, 2, (0.73, 0.77, 0.67, 0.62), False),

        # Week 3 ago — v1.1.x tuning
        (3, 0, 0, (0.77, 0.80, 0.72, 0.68), False),
        (3, 1, 3, (0.79, 0.82, 0.74, 0.70), False),
        (3, 2, 1, (0.63, 0.69, 0.60, 0.55), True),   # regression run
        (3, 4, 0, (0.80, 0.83, 0.75, 0.71), False),
        (3, 6, 2, (0.78, 0.81, 0.73, 0.69), False),

        # Week 2 ago — v1.2.x: strong results
        (2, 0, 0, (0.83, 0.86, 0.79, 0.75), False),
        (2, 1, 1, (0.85, 0.87, 0.81, 0.77), False),
        (2, 3, 0, (0.84, 0.87, 0.80, 0.76), False),
        (2, 4, 3, (0.86, 0.88, 0.82, 0.78), False),
        (2, 6, 2, (0.82, 0.85, 0.78, 0.74), False),

        # Week 1 ago — v1.2.x refinement
        (1, 0, 0, (0.87, 0.89, 0.83, 0.80), False),
        (1, 2, 1, (0.88, 0.90, 0.85, 0.81), False),
        (1, 3, 0, (0.86, 0.89, 0.82, 0.79), False),
        (1, 5, 3, (0.90, 0.91, 0.87, 0.83), False),

        # This week — v1.3.x: latest, best scores
        (0, 1, 0, (0.91, 0.92, 0.88, 0.85), False),
        (0, 2, 1, (0.92, 0.93, 0.89, 0.86), False),
        (0, 3, 0, (0.78, 0.82, 0.75, 0.70), True),   # regression (chunk size change)
        (0, 4, 3, (0.93, 0.94, 0.90, 0.87), False),
        (0, 5, 2, (0.92, 0.93, 0.89, 0.86), False),
        (0, 6, 4, (0.91, 0.93, 0.88, 0.85), False),
    ]

    # Version tag series
    version_series = {
        6: "v1.0", 5: "v1.0", 4: "v1.1", 3: "v1.1",
        2: "v1.2", 1: "v1.2", 0: "v1.3",
    }
    run_counter = {k: 0 for k in version_series}

    prev_scores: dict[int, tuple] = {}

    for i, (week_ago, day_in_week, pipe_idx, base_scores, is_regression) in enumerate(configs):
        created_at = now - timedelta(weeks=week_ago, days=day_in_week, hours=random.randint(0, 12))
        completed_at = created_at + timedelta(minutes=random.randint(3, 8))

        run_counter[week_ago] += 1
        version = f"{version_series[week_ago]}.{run_counter[week_ago]}-eval"
        pipeline_name, model_name = PIPELINES[pipe_idx]

        f, r, p, c = _make_scores(*base_scores)

        regression_details: dict = {}
        has_regression = False
        threshold = 0.10
        if pipe_idx in prev_scores:
            pf, pr, pp, pc = prev_scores[pipe_idx]
            for metric_name, (curr_val, prev_val) in [
                ("faithfulness", (f, pf)),
                ("answer_relevancy", (r, pr)),
                ("context_precision", (p, pp)),
                ("context_recall", (c, pc)),
            ]:
                delta = curr_val - prev_val
                regression_details[metric_name] = {
                    "previous": round(prev_val, 4),
                    "current": round(curr_val, 4),
                    "delta": round(delta, 4),
                    "threshold": threshold,
                    "is_regression": is_regression and delta <= -threshold,
                }
            has_regression = any(v["is_regression"] for v in regression_details.values())

        prev_scores[pipe_idx] = (f, r, p, c)

        # Per-case scores (10 cases, individual variation)
        cases = []
        for q_idx in range(10):
            case_f, case_r, case_p, case_c = _make_scores(f, r, p, c, noise=0.12)
            cases.append({
                "question": QUESTIONS[q_idx],
                "answer": ANSWERS[q_idx],
                "ground_truth": GROUND_TRUTHS[q_idx],
                "contexts": [
                    f"Context chunk from {pipeline_name} retrieval — query #{q_idx+1}",
                    f"Secondary context retrieved with score {round(0.7 + random.random() * 0.25, 3)}",
                ],
                "faithfulness_score": round(case_f, 4),
                "answer_relevancy_score": round(case_r, 4),
                "context_precision_score": round(case_p, 4),
                "context_recall_score": round(case_c, 4),
                "created_at": created_at + timedelta(seconds=q_idx * 30),
            })

        runs.append({
            "id": str(uuid.uuid4()),
            "version_tag": version,
            "pipeline_name": pipeline_name,
            "status": "completed",
            "created_at": created_at,
            "completed_at": completed_at,
            "total_cases": 10,
            "avg_faithfulness": round(f, 4),
            "avg_answer_relevancy": round(r, 4),
            "avg_context_precision": round(p, 4),
            "avg_context_recall": round(c, 4),
            "run_metadata": {
                "model": model_name,
                "dataset": "ai_ml_benchmark_v2",
                "chunk_size": [256, 512, 1024][pipe_idx % 3],
                "strategy": "mmr" if pipe_idx % 2 == 0 else "similarity",
                "source": "seed_data",
            },
            "has_regression": has_regression,
            "regression_details": regression_details,
            "cases": cases,
        })

    return runs


def seed_eval_data_sync() -> None:
    """Insert seed eval runs if the DB is currently empty. Idempotent."""
    from sqlalchemy import create_engine, select, func
    from sqlalchemy.orm import sessionmaker
    from backend.database import _get_sync_engine
    from backend.models import EvalRun, EvalCase

    engine = _get_sync_engine()
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as sess:
        count = sess.execute(select(func.count(EvalRun.id))).scalar_one()
        if count >= 25:
            logger.info(f"Eval seed skipped — {count} runs already exist (sufficient data)")
            return

    logger.info("Seeding eval data (30+ runs with realistic scores)...")
    runs_data = _run_data({})

    with Session() as sess:
        for rd in runs_data:
            run_id = uuid.UUID(rd["id"])
            run = EvalRun(
                id=run_id,
                version_tag=rd["version_tag"],
                pipeline_name=rd["pipeline_name"],
                status=rd["status"],
                created_at=rd["created_at"],
                completed_at=rd["completed_at"],
                total_cases=rd["total_cases"],
                avg_faithfulness=rd["avg_faithfulness"],
                avg_answer_relevancy=rd["avg_answer_relevancy"],
                avg_context_precision=rd["avg_context_precision"],
                avg_context_recall=rd["avg_context_recall"],
                run_metadata=rd["run_metadata"],
                has_regression=rd["has_regression"],
                regression_details=rd["regression_details"],
            )
            sess.add(run)
            for cd in rd["cases"]:
                sess.add(EvalCase(
                    eval_run_id=run_id,
                    question=cd["question"],
                    answer=cd["answer"],
                    ground_truth=cd["ground_truth"],
                    contexts=cd["contexts"],
                    faithfulness_score=cd["faithfulness_score"],
                    answer_relevancy_score=cd["answer_relevancy_score"],
                    context_precision_score=cd["context_precision_score"],
                    context_recall_score=cd["context_recall_score"],
                    created_at=cd["created_at"],
                ))
        sess.commit()

    logger.info(f"Seeded {len(runs_data)} eval runs with {len(runs_data) * 10} cases")
