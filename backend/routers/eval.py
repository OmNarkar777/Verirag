"""routers/eval.py â€” Evaluation endpoints with rate limiting and regression detection."""
import asyncio
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
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


@router.post("/run/sample", response_model=EvalRunResponse, status_code=200)
async def run_sample_eval(
    db: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> EvalRunResponse:
    """
    Run a 3-case RAGAS evaluation synchronously using the built-in sample dataset.

    Uses Groq as the LLM judge (Faithfulness, AnswerRelevancy, ContextPrecision,
    ContextRecall). Embeddings use the local TF-IDF fallback when HF_TOKEN is absent.
    Runs inline (no background task) so Vercel's Lambda keeps the connection open
    until completion. Typical duration: 20-45 seconds.

    Each invocation generates a unique timestamped version tag so every button
    click creates a distinct run visible on the Dashboard trend chart.
    """
    from datetime import datetime, timezone
    from backend.evaluator.dataset_builder import get_sample_test_cases

    # Unique version tag per invocation — visible as a distinct chart point
    ts = datetime.now(timezone.utc).strftime("%m%d%H%M")
    version_tag = f"v1.{ts[:4]}.{ts[4:]}-sample"

    # 3 cases keeps the evaluation under Vercel's 60s Lambda limit
    test_cases = get_sample_test_cases()[:3]

    try:
        eval_run_id = await service.start_eval_run(
            db=db,
            version_tag=version_tag,
            pipeline_name="sample-ragas-eval",
            test_cases=test_cases,
            metadata={"dataset": "sample_3_ai_ml_cases", "source": "builtin"},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Run RAGAS synchronously — the HTTP request blocks until complete.
    # Vercel keeps the Lambda alive for maxDuration (60s) after response starts.
    try:
        await service.runner.run_evaluation(eval_run_id, test_cases)
    except Exception as e:
        logger.error(f"Sample RAGAS eval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAGAS evaluation failed: {str(e)[:300]}",
        )

    return EvalRunResponse(
        eval_run_id=eval_run_id,
        version_tag=version_tag,
        status="completed",
        message=f"Sample evaluation complete ({len(test_cases)} cases with real RAGAS scores). Dashboard updated.",
    )


@router.get("/runs", response_model=list[EvalRunWithRegression])
async def list_eval_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[EvalRunWithRegression]:
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit).offset(offset)
    )
    runs = result.scalars().all()
    return [_run_to_regression_schema(r) for r in runs]


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