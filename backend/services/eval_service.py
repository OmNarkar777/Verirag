"""
services/eval_service.py — Business logic layer for evaluation operations.

WHY A SERVICE LAYER:
Routers handle HTTP concerns (request parsing, status codes, response formatting).
Services handle business logic (querying DB, orchestrating eval runs).
This separation enables:
- Testing service logic without HTTP context
- Reusing logic across routes (e.g., both REST and future gRPC)
- Clean dependency injection without route handler bloat
"""

import uuid

from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluator.dataset_builder import get_sample_test_cases
from backend.evaluator.ragas_runner import RagasRunner, get_ragas_runner
from backend.models import EvalCase, EvalRun
from backend.schemas import (
    EvalCaseResult,
    EvalRunDetail,
    EvalRunSummary,
    MetricScores,
    PaginatedCases,
    TestCaseInput,
)


class EvalService:
    """
    Business logic for creating, running, and querying evaluations.

    All DB operations are async — never blocks the event loop.
    RAGAS execution is delegated to RagasRunner (runs in thread pool).

    The RagasRunner is initialised lazily so that read-only endpoints
    (list runs, get run detail) work even when GROQ_API_KEY is absent.
    """

    def __init__(self, runner: RagasRunner | None = None):
        self._runner = runner  # may be None; resolved on first write operation

    @property
    def runner(self) -> RagasRunner:
        if self._runner is None:
            self._runner = get_ragas_runner()
        return self._runner

    async def start_eval_run(
        self,
        db: AsyncSession,
        version_tag: str,
        pipeline_name: str,
        test_cases: list[TestCaseInput],
        metadata: dict,
    ) -> uuid.UUID:
        """
        Create an EvalRun record and return its ID immediately.
        
        The caller should then launch run_evaluation() as a BackgroundTask.
        This two-step pattern allows the API to return 202 Accepted instantly
        while the 5-minute eval runs asynchronously.
        """
        run_id = await self.runner.create_eval_run(
            version_tag=version_tag,
            pipeline_name=pipeline_name,
            total_cases=len(test_cases),
            metadata=metadata,
        )
        logger.info(f"EvalRun created | id={run_id}")
        return run_id

    async def execute_evaluation(
        self,
        eval_run_id: uuid.UUID,
        test_cases: list[TestCaseInput],
    ) -> None:
        """
        Execute RAGAS evaluation (intended to run as BackgroundTask).
        
        Errors are caught and stored in the EvalRun record — never propagated
        to crash the background task silently.
        """
        try:
            await self.runner.run_evaluation(
                eval_run_id=eval_run_id,
                test_cases=test_cases,
            )
        except Exception as e:
            # Runner already handles marking the run as failed in PostgreSQL.
            # We just log here for background task visibility.
            logger.error(
                f"Background eval task failed | "
                f"run_id={eval_run_id} | error={e}"
            )

    async def list_eval_runs(
        self,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EvalRunSummary]:
        """
        List all eval runs, most recent first.
        Returns lightweight summaries (no per-case data loaded).
        """
        result = await db.execute(
            select(EvalRun)
            .order_by(desc(EvalRun.created_at))
            .limit(limit)
            .offset(offset)
        )
        runs = result.scalars().all()
        return [_run_to_summary(run) for run in runs]

    async def get_eval_run(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
    ) -> EvalRunDetail | None:
        """
        Get full eval run detail including all per-case scores.
        
        Uses selectinload for cases — issues a single IN query rather than N+1.
        For 1000s of cases, consider pagination here too.
        """
        from sqlalchemy.orm import selectinload

        result = await db.execute(
            select(EvalRun)
            .options(selectinload(EvalRun.cases))
            .where(EvalRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if not run:
            return None
        return _run_to_detail(run)

    async def get_eval_cases_paginated(
        self,
        db: AsyncSession,
        run_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedCases | None:
        """
        Paginated case results for large eval runs.
        
        WHY PAGINATION:
        A production eval might have 500+ test cases. Loading all at once
        kills the UI and wastes memory. Pagination is a production requirement.
        """
        # Verify the run exists
        run_result = await db.execute(
            select(EvalRun).where(EvalRun.id == run_id)
        )
        if not run_result.scalar_one_or_none():
            return None

        # Count total cases
        count_result = await db.execute(
            select(func.count(EvalCase.id)).where(EvalCase.eval_run_id == run_id)
        )
        total = count_result.scalar_one()

        # Fetch paginated cases
        cases_result = await db.execute(
            select(EvalCase)
            .where(EvalCase.eval_run_id == run_id)
            .order_by(EvalCase.created_at)
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
        cases = cases_result.scalars().all()

        return PaginatedCases(
            total=total,
            page=page,
            page_size=page_size,
            cases=[EvalCaseResult.model_validate(case) for case in cases],
        )

    async def run_sample_evaluation(
        self,
        db: AsyncSession,
        version_tag: str = "v0.0.1-sample",
    ) -> uuid.UUID:
        """
        Convenience method to run evaluation on the built-in sample dataset.
        Useful for smoke testing the pipeline without providing test data.
        """
        test_cases = get_sample_test_cases()
        run_id = await self.start_eval_run(
            db=db,
            version_tag=version_tag,
            pipeline_name="sample-pipeline",
            test_cases=test_cases,
            metadata={"dataset": "sample_10_ai_ml_cases"},
        )
        return run_id


def _run_to_summary(run: EvalRun) -> EvalRunSummary:
    """Convert ORM EvalRun to EvalRunSummary schema."""
    scores = MetricScores(
        faithfulness=run.avg_faithfulness,
        answer_relevancy=run.avg_answer_relevancy,
        context_precision=run.avg_context_precision,
        context_recall=run.avg_context_recall,
    ) if run.status == "completed" else None

    return EvalRunSummary(
        id=run.id,
        version_tag=run.version_tag,
        pipeline_name=run.pipeline_name,
        status=run.status,
        created_at=run.created_at,
        completed_at=run.completed_at,
        total_cases=run.total_cases,
        scores=scores,
    )


def _run_to_detail(run: EvalRun) -> EvalRunDetail:
    """Convert ORM EvalRun + cases to EvalRunDetail schema."""
    summary = _run_to_summary(run)
    cases = [EvalCaseResult.model_validate(case) for case in run.cases]

    return EvalRunDetail(
        **summary.model_dump(),
        cases=cases,
        run_metadata=run.run_metadata or {},
        error_message=run.error_message,
    )


def get_eval_service() -> EvalService:
    """FastAPI dependency for EvalService."""
    return EvalService()
