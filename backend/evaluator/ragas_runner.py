"""evaluator/ragas_runner.py â€" Core RAGAS evaluation engine with regression detection."""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from loguru import logger
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from backend.config import get_settings
from backend.database import get_db_context
from backend.evaluator.dataset_builder import build_ragas_dataset
from backend.models import EvalCase, EvalRun
from backend.schemas import TestCaseInput

settings = get_settings()


class HFInferenceEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper.

    When HF_TOKEN is set: calls HuggingFace Inference API (sentence-transformers/all-MiniLM-L6-v2).
    When HF_TOKEN is absent (default on Vercel): uses the local TF-IDF hash embedder so
    RAGAS evaluation works fully offline. Both paths produce 384-dim normalised vectors;
    retrieval quality differs but all metrics compute correctly.
    """
    def __init__(self):
        self._client = InferenceClient(token=settings.hf_token or None) if settings.hf_token else None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from backend.rag.vectorstore import _fallback_embed
        if not settings.hf_token or self._client is None:
            return _fallback_embed(texts)
        try:
            response = self._client.feature_extraction(
                text=texts,
                model=settings.embedding_model,
            )
            if hasattr(response, "tolist"):
                result = response.tolist()
                if result and not isinstance(result[0], list):
                    return [result]
                return result
            return [list(vec) for vec in response]
        except Exception as e:
            logger.warning(f"HF embeddings failed ({str(e)[:80]}); using local TF-IDF fallback")
            return _fallback_embed(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class RagasRunner:
    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required for RAGAS evaluation. "
                "Set it in your Vercel environment variables."
            )
        self.judge_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.0,
        )
        self.embeddings = HFInferenceEmbeddings()
        logger.info(f"RagasRunner initialized | judge_model={settings.groq_model}")

    def _configure_metrics(self) -> list:
        """
        Inject LLM into RAGAS metrics via LangchainLLMWrapper.
        Direct assignment of a raw ChatGroq fails in RAGAS 0.1.9+;
        wrapping is the correct integration pattern.
        """
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            wrapped_llm = LangchainLLMWrapper(self.judge_llm)
            wrapped_emb = LangchainEmbeddingsWrapper(self.embeddings)
        except ImportError:
            # Older RAGAS version â€" direct assignment works
            wrapped_llm = self.judge_llm
            wrapped_emb = self.embeddings

        faithfulness = Faithfulness()
        faithfulness.llm = wrapped_llm

        answer_relevancy = AnswerRelevancy()
        answer_relevancy.llm = wrapped_llm
        answer_relevancy.embeddings = wrapped_emb

        context_precision = ContextPrecision()
        context_precision.llm = wrapped_llm

        context_recall = ContextRecall()
        context_recall.llm = wrapped_llm

        return [faithfulness, answer_relevancy, context_precision, context_recall]

    async def create_eval_run(self, version_tag, pipeline_name, total_cases, metadata) -> uuid.UUID:
        async with get_db_context() as db:
            run = EvalRun(
                version_tag=version_tag, pipeline_name=pipeline_name,
                status="running", total_cases=total_cases,
                run_metadata={
                    "groq_model": settings.groq_model,
                    "embedding_model": settings.embedding_model,
                    "retrieval_top_k": settings.retrieval_top_k,
                    **metadata,
                },
            )
            db.add(run)
            await db.flush()
            run_id = run.id
            logger.info(f"Created EvalRun | id={run_id} | version={version_tag}")
            return run_id

    async def run_evaluation(self, eval_run_id: uuid.UUID, test_cases: list[TestCaseInput]) -> None:
        logger.info(f"Starting RAGAS evaluation | run_id={eval_run_id} | cases={len(test_cases)}")
        try:
            dataset = build_ragas_dataset(test_cases)
            metrics = self._configure_metrics()

            logger.info("Running RAGAS evaluate() in thread pool...")

            def _ragas_in_thread():
                # Python 3.12: asyncio.get_event_loop() raises RuntimeError in non-main
                # threads without a running loop. RAGAS internals call this, so we must
                # provide a fresh event loop in the worker thread before calling evaluate().
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return evaluate(
                        dataset=dataset,
                        metrics=metrics,
                        raise_exceptions=False,  # return NaN rather than crash on metric failure
                    )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            result = await asyncio.get_running_loop().run_in_executor(None, _ragas_in_thread)

            scores_df = result.to_pandas()
            logger.info(f"RAGAS evaluate() complete | shape={scores_df.shape}")

            await self._persist_cases(eval_run_id, test_cases, scores_df)
            await self._complete_eval_run(eval_run_id, scores_df)

            # Regression detection â€" non-fatal
            try:
                await self._run_regression_check(eval_run_id)
            except Exception as e:
                logger.warning(f"Regression check failed (non-fatal): {e}")

            # LangSmith tagging â€" non-fatal
            try:
                await self._tag_langsmith(eval_run_id)
            except Exception as e:
                logger.warning(f"LangSmith tagging failed (non-fatal): {e}")

        except Exception as e:
            logger.error(f"RAGAS evaluation failed | run_id={eval_run_id} | error={e}")
            await self._fail_eval_run(eval_run_id, str(e))
            raise

    async def _persist_cases(self, eval_run_id, test_cases, scores_df) -> None:
        async with get_db_context() as db:
            cases = []
            for i, tc in enumerate(test_cases):
                row = scores_df.iloc[i] if i < len(scores_df) else None

                def safe_score(col: str) -> float | None:
                    if row is None or col not in scores_df.columns:
                        return None
                    val = row[col]
                    try:
                        import math
                        f = float(val)
                        return None if math.isnan(f) else f
                    except (TypeError, ValueError):
                        return None

                cases.append(EvalCase(
                    eval_run_id=eval_run_id,
                    question=tc.question, answer=tc.answer,
                    contexts=tc.contexts, ground_truth=tc.ground_truth,
                    faithfulness_score=safe_score("faithfulness"),
                    answer_relevancy_score=safe_score("answer_relevancy"),
                    context_precision_score=safe_score("context_precision"),
                    context_recall_score=safe_score("context_recall"),
                ))
            db.add_all(cases)
            logger.info(f"Persisted {len(cases)} eval cases | run_id={eval_run_id}")

    async def _complete_eval_run(self, eval_run_id, scores_df) -> None:
        import numpy as np
        def mean_score(col):
            if col not in scores_df.columns:
                return None
            vals = scores_df[col].dropna()
            return float(np.mean(vals)) if len(vals) > 0 else None

        async with get_db_context() as db:
            from sqlalchemy import select
            result = await db.execute(select(EvalRun).where(EvalRun.id == eval_run_id))
            run = result.scalar_one_or_none()
            if not run:
                return
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.avg_faithfulness = mean_score("faithfulness")
            run.avg_answer_relevancy = mean_score("answer_relevancy")
            run.avg_context_precision = mean_score("context_precision")
            run.avg_context_recall = mean_score("context_recall")

            from backend.services.langsmith_service import get_langsmith_service
            ls = get_langsmith_service()
            run.langsmith_run_url = ls.get_project_url()

            logger.info(
                f"EvalRun completed | id={eval_run_id} | "
                f"faithfulness={run.avg_faithfulness:.3f if run.avg_faithfulness else 'N/A'}"
            )

    async def _run_regression_check(self, eval_run_id: uuid.UUID) -> None:
        from backend.services.regression_service import detect_and_store_regressions
        from backend.database import get_db_context
        async with get_db_context() as db:
            try:
                await detect_and_store_regressions(db, eval_run_id)
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def _tag_langsmith(self, eval_run_id: uuid.UUID) -> None:
        async with get_db_context() as db:
            from sqlalchemy import select
            result = await db.execute(select(EvalRun).where(EvalRun.id == eval_run_id))
            run = result.scalar_one_or_none()
            if not run:
                return
        from backend.services.langsmith_service import get_langsmith_service
        ls = get_langsmith_service()
        recent = ls.list_recent_traces(limit=5)
        tags = [f"eval_run:{str(eval_run_id)[:8]}", run.version_tag, run.pipeline_name]
        for trace in recent:
            ls.tag_run(trace["run_id"], tags)

    async def _fail_eval_run(self, eval_run_id: uuid.UUID, error: str) -> None:
        async with get_db_context() as db:
            from sqlalchemy import select
            result = await db.execute(select(EvalRun).where(EvalRun.id == eval_run_id))
            run = result.scalar_one_or_none()
            if run:
                run.status = "failed"
                run.completed_at = datetime.now(timezone.utc)
                run.error_message = error[:1000]
                logger.warning(f"EvalRun marked failed | id={eval_run_id}")


_runner: RagasRunner | None = None


def get_ragas_runner() -> RagasRunner:
    global _runner
    if _runner is None:
        _runner = RagasRunner()
    return _runner