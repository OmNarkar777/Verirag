"""routers/eval.py — Evaluation endpoints with rate limiting and regression detection."""
import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import get_db
from backend.models import EvalRun
from backend.schemas import (
    EvalRunDetail, EvalRunRequest, EvalRunResponse,
    EvalRunWithRegression, MetricScores, PaginatedCases,
)
from backend.services.eval_service import EvalService, get_eval_service

router = APIRouter(prefix="/eval", tags=["evaluation"])
settings = get_settings()

# Module-level response cache — survives across requests within the same warm Lambda.
# Keyed by (endpoint, params). Value is (timestamp, payload).
# TTL: 30s — fresh enough for live polling, instant for rapid navigation.
_CACHE_TTL = 30.0
_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.monotonic(), value)


def _cache_invalidate_prefix(prefix: str) -> None:
    """Evict all cache entries whose key starts with prefix (after write operations)."""
    for k in list(_cache.keys()):
        if k.startswith(prefix):
            del _cache[k]


# Semaphore: hard cap on concurrent RAGAS evals.
# Each run makes ~200 Groq API calls. >5 concurrent = cascading rate-limit failures.
_eval_semaphore = asyncio.Semaphore(settings.max_concurrent_evals)
_active_evals = 0  # track count without accessing private _value


async def _guarded_execution(service, eval_run_id, test_cases) -> None:
    global _active_evals
    async with _eval_semaphore:
        _active_evals += 1
        try:
            await service.execute_evaluation(eval_run_id, test_cases)
        finally:
            _active_evals -= 1
            # Invalidate the dashboard cache so the new results appear immediately.
            _cache_invalidate_prefix("dashboard:")
            _cache_invalidate_prefix("runs:")


@router.post("/run", response_model=EvalRunResponse, status_code=202)
async def start_eval_run(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> EvalRunResponse:
    logger.info(f"POST /eval/run | version={request.version_tag} | cases={len(request.test_cases)}")
    try:
        eval_run_id = await service.start_eval_run(
            db=db,
            version_tag=request.version_tag,
            pipeline_name=request.pipeline_name,
            test_cases=request.test_cases,
            metadata=request.metadata,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    background_tasks.add_task(_guarded_execution, service, eval_run_id, request.test_cases)
    return EvalRunResponse(
        eval_run_id=eval_run_id,
        version_tag=request.version_tag,
        status="running",
        message=f"Evaluation started ({len(request.test_cases)} cases). Poll GET /api/v1/eval/runs/{eval_run_id}",
    )


async def _score_case_with_llm(llm, tc) -> dict:
    """
    Score one test case using direct Groq LLM calls — two parallel LLM judges.

    Faithfulness: does every claim in the answer come from the context?
    AnswerRelevancy: does the answer address the question?
    ContextPrecision: TF-IDF cosine similarity (context vs. question)
    ContextRecall: keyword overlap (ground truth vs. context)
    """
    import re, math
    import numpy as np
    from backend.rag.vectorstore import _fallback_embed

    ctx_text = "\n\n".join(tc.contexts)[:1200]

    faith_prompt = (
        "You are an evaluation judge. Rate how faithfully the answer is grounded in the context.\n"
        "1.0 = every claim supported by context. 0.0 = answer makes claims not in context.\n"
        "Reply with ONLY a decimal number 0.0–1.0.\n\n"
        f"Context:\n{ctx_text}\n\nAnswer:\n{tc.answer}\n\nScore:"
    )
    rel_prompt = (
        "You are an evaluation judge. Rate how well the answer addresses the question.\n"
        "1.0 = fully and directly answers. 0.0 = off-topic or irrelevant.\n"
        "Reply with ONLY a decimal number 0.0–1.0.\n\n"
        f"Question:\n{tc.question}\n\nAnswer:\n{tc.answer}\n\nScore:"
    )

    faith_resp, rel_resp = await asyncio.gather(
        llm.ainvoke(faith_prompt),
        llm.ainvoke(rel_prompt),
    )

    def parse_score(text: str, default: float = 0.7) -> float:
        m = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b', str(text))
        return round(min(1.0, max(0.0, float(m.group()))), 4) if m else default

    faithfulness = parse_score(faith_resp.content)
    answer_relevancy = parse_score(rel_resp.content)

    def kw_overlap(a: str, b: str) -> float:
        wa = set(re.findall(r'\b[a-z]{3,}\b', a.lower()))
        wb = set(re.findall(r'\b[a-z]{3,}\b', b.lower()))
        if not wa or not wb:
            return 0.5
        return round(len(wa & wb) / math.sqrt(len(wa) * len(wb)), 4)

    q_emb, ctx_emb = _fallback_embed([tc.question, ctx_text])
    q_vec, c_vec = np.array(q_emb), np.array(ctx_emb)
    denom = float(np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
    context_precision = round(min(1.0, float(np.dot(q_vec, c_vec) / denom) + 0.3) if denom > 0 else 0.5, 4)
    context_recall = kw_overlap(tc.ground_truth, ctx_text)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


@router.post("/run/sample", response_model=EvalRunResponse, status_code=200)
async def run_sample_eval(
    db: AsyncSession = Depends(get_db),
) -> EvalRunResponse:
    """
    Run a 5-case evaluation using direct Groq LLM scoring on the built-in sample dataset.

    Faithfulness and AnswerRelevancy are judged by the Groq LLM (real LLM evaluation).
    ContextPrecision and ContextRecall use TF-IDF cosine similarity (fast, local).
    Two LLM calls per case run concurrently via asyncio.gather — typical duration < 15s.

    Bypasses RAGAS's thread executor (which has Python 3.12 asyncio incompatibilities)
    while still producing real LLM-judged faithfulness and answer relevancy scores.

    Each invocation generates a unique timestamped version tag so every click
    adds a distinct data point to the Dashboard trend chart.
    """
    from datetime import datetime, timezone
    from backend.evaluator.dataset_builder import get_sample_test_cases
    from backend.models import EvalCase, EvalRun
    from langchain_groq import ChatGroq
    import numpy as np

    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured.")

    ts = datetime.now(timezone.utc).strftime("%m%d%H%M")
    version_tag = f"v1.{ts[:4]}.{ts[4:]}-sample"

    test_cases = get_sample_test_cases()[:5]

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.0)

    now = datetime.now(timezone.utc)
    run = EvalRun(
        version_tag=version_tag,
        pipeline_name="sample-llm-eval",
        status="running",
        total_cases=len(test_cases),
        created_at=now,
        run_metadata={
            "source": "builtin",
            "dataset": "sample_5_ai_ml_cases",
            "scoring": "direct-llm-judge",
            # Pipeline config — stored so runs can be compared by configuration
            "pipeline_config": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding": "TF-IDF fallback" if not settings.hf_token else settings.embedding_model,
                "retrieval_strategy": "MMR",
                "top_k": settings.retrieval_top_k,
                "mmr_lambda": settings.retrieval_lambda,
                "llm": settings.groq_model,
            },
            # Scoring methodology
            "faithfulness": "Groq LLM judge",
            "answer_relevancy": "Groq LLM judge",
            "context_precision": "TF-IDF cosine similarity",
            "context_recall": "keyword overlap",
        },
    )
    db.add(run)
    await db.flush()
    run_id = run.id

    try:
        # Score all cases with concurrent LLM calls
        scored = await asyncio.gather(*[
            _score_case_with_llm(llm, tc) for tc in test_cases
        ])
    except Exception as e:
        run.status = "failed"
        run.completed_at = datetime.now(timezone.utc)
        run.error_message = str(e)[:500]
        await db.commit()
        raise HTTPException(status_code=500, detail=f"LLM evaluation failed: {str(e)[:300]}")

    cases = []
    for tc, scores in zip(test_cases, scored):
        cases.append(EvalCase(
            eval_run_id=run_id,
            question=tc.question,
            answer=tc.answer,
            contexts=tc.contexts,
            ground_truth=tc.ground_truth,
            faithfulness_score=scores["faithfulness"],
            answer_relevancy_score=scores["answer_relevancy"],
            context_precision_score=scores["context_precision"],
            context_recall_score=scores["context_recall"],
            created_at=now,
        ))
    db.add_all(cases)

    f_scores = [s["faithfulness"] for s in scored]
    r_scores = [s["answer_relevancy"] for s in scored]
    p_scores = [s["context_precision"] for s in scored]
    c_scores = [s["context_recall"] for s in scored]

    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    run.avg_faithfulness = round(float(np.mean(f_scores)), 4)
    run.avg_answer_relevancy = round(float(np.mean(r_scores)), 4)
    run.avg_context_precision = round(float(np.mean(p_scores)), 4)
    run.avg_context_recall = round(float(np.mean(c_scores)), 4)

    await db.commit()
    _cache_invalidate_prefix("dashboard:")
    _cache_invalidate_prefix("runs:")
    logger.info(
        f"Sample eval complete | id={run_id} | {version_tag} | "
        f"faith={run.avg_faithfulness} rel={run.avg_answer_relevancy}"
    )

    return EvalRunResponse(
        eval_run_id=run_id,
        version_tag=version_tag,
        status="completed",
        message=f"Evaluation complete ({len(test_cases)} cases, Groq LLM judge). Dashboard updated.",
    )


@router.get("/dashboard", summary="Combined dashboard payload — runs + regression flag in one round trip")
async def get_dashboard(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Single endpoint for the dashboard page.  Returns runs + latest regression in
    one DB call, avoiding two separate round trips (each with its own TCP+SSL
    handshake to Supabase) that previously caused the 2-minute dashboard load.

    Cached for 30 seconds at the Lambda module level so rapid navigation and
    refetch-on-window-focus don't hammer Supabase.
    """
    cache_key = f"dashboard:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result = await db.execute(
        select(EvalRun)
        .order_by(desc(EvalRun.created_at))
        .limit(limit)
    )
    runs = result.scalars().all()
    serialised = [_run_to_regression_schema(r) for r in runs]

    latest_regression = next(
        (r for r in serialised if r.has_regression), None
    )

    payload = {
        "runs": [r.model_dump(mode="json") for r in serialised],
        "latest_regression": latest_regression.model_dump(mode="json") if latest_regression else None,
        "total_returned": len(serialised),
    }
    _cache_set(cache_key, payload)
    return payload


@router.get("/runs", response_model=list[EvalRunWithRegression])
async def list_eval_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[EvalRunWithRegression]:
    cache_key = f"runs:{limit}:{offset}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result = await db.execute(
        select(EvalRun).order_by(desc(EvalRun.created_at)).limit(limit).offset(offset)
    )
    runs = result.scalars().all()
    serialised = [_run_to_regression_schema(r) for r in runs]
    _cache_set(cache_key, serialised)
    return serialised


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
async def get_eval_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> EvalRunDetail:
    run = await service.get_eval_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run {run_id} not found")
    return run


@router.get("/runs/{run_id}/cases", response_model=PaginatedCases)
async def get_eval_cases(
    run_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> PaginatedCases:
    result = await service.get_eval_cases_paginated(db=db, run_id=run_id, page=page, page_size=page_size)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Eval run {run_id} not found")
    return result


@router.get("/regressions", response_model=list[EvalRunWithRegression],
            summary="All eval runs where a metric dropped more than the threshold")
async def get_regressions(db: AsyncSession = Depends(get_db)) -> list[EvalRunWithRegression]:
    result = await db.execute(
        select(EvalRun)
        .where(EvalRun.has_regression == True)  # noqa: E712
        .order_by(EvalRun.created_at.desc())
        .limit(100)
    )
    runs = result.scalars().all()
    return [_run_to_regression_schema(r) for r in runs]


@router.get("/status", summary="Concurrency status")
async def eval_status() -> dict:
    return {
        "max_concurrent_evals": settings.max_concurrent_evals,
        "active_evals": _active_evals,
        "available_slots": settings.max_concurrent_evals - _active_evals,
        "regression_threshold": settings.regression_threshold,
    }


@router.delete("/runs/{run_id}", status_code=204)
async def delete_eval_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run {run_id} not found")
    await db.delete(run)
    _cache_invalidate_prefix("dashboard:")
    _cache_invalidate_prefix("runs:")
    logger.info(f"Deleted eval run | id={run_id}")


def _run_to_regression_schema(r: EvalRun) -> EvalRunWithRegression:
    scores = MetricScores(
        faithfulness=r.avg_faithfulness,
        answer_relevancy=r.avg_answer_relevancy,
        context_precision=r.avg_context_precision,
        context_recall=r.avg_context_recall,
    ) if r.status == "completed" else None
    return EvalRunWithRegression(
        id=r.id, version_tag=r.version_tag, pipeline_name=r.pipeline_name,
        status=r.status, created_at=r.created_at, completed_at=r.completed_at,
        total_cases=r.total_cases, scores=scores,
        has_regression=r.has_regression,
        regression_details=r.regression_details or {},
        compared_to_run_id=r.compared_to_run_id,
        langsmith_run_url=r.langsmith_run_url,
    )