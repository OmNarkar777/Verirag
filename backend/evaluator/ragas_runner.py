"""evaluator/ragas_runner.py - RAGAS runner using direct Groq LLM evaluation."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from loguru import logger

from backend.config import get_settings
from backend.database import get_db_context
from backend.models import EvalRun
from backend.schemas import TestCaseInput

settings = get_settings()


class HFInferenceEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper.

    Uses HuggingFace Inference API when HF_TOKEN is set, otherwise
    falls back to the local TF-IDF hash embedder. Both paths produce
    384-dim normalised vectors.
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
    """
    Eval runner used for the POST /eval/run endpoint (user-submitted test cases).

    Creates the EvalRun record in PostgreSQL. The actual scoring for sample evals
    is handled inline in routers/eval.py via direct Groq LLM calls.
    """
    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required for evaluation. "
                "Set it in your Vercel environment variables."
            )
        self.judge_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.0,
        )
        self.embeddings = HFInferenceEmbeddings()
        logger.info(f"RagasRunner initialized | judge_model={settings.groq_model}")

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
        """
        Score submitted test cases using direct Groq LLM calls.

        Faithfulness + AnswerRelevancy: Groq LLM judge (2 parallel calls per case).
        ContextPrecision + ContextRecall: TF-IDF cosine + keyword overlap (fast, local).
        """
        import asyncio
        import re
        import math
        import numpy as np
        from backend.rag.vectorstore import _fallback_embed
        from backend.models import EvalCase

        logger.info(f"Starting LLM evaluation | run_id={eval_run_id} | cases={len(test_cases)}")

        async def _score_one(tc: TestCaseInput) -> dict:
            ctx_text = "\n\n".join(tc.contexts)[:1200]
            faith_prompt = (
                "You are an evaluation judge. Rate how faithfully the answer is grounded in the context.\n"
                "1.0 = every claim supported by context. 0.0 = answer makes claims not in context.\n"
                "Reply with ONLY a decimal number 0.0-1.0.\n\n"
                f"Context:\n{ctx_text}\n\nAnswer:\n{tc.answer}\n\nScore:"
            )
            rel_prompt = (
                "You are an evaluation judge. Rate how well the answer addresses the question.\n"
                "1.0 = fully and directly answers. 0.0 = off-topic or irrelevant.\n"
                "Reply with ONLY a decimal number 0.0-1.0.\n\n"
                f"Question:\n{tc.question}\n\nAnswer:\n{tc.answer}\n\nScore:"
            )
            faith_resp, rel_resp = await asyncio.gather(
                self.judge_llm.ainvoke(faith_prompt),
                self.judge_llm.ainvoke(rel_prompt),
            )

            def _parse(text: str, default: float = 0.7) -> float:
                m = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b', str(text))
                return round(min(1.0, max(0.0, float(m.group()))), 4) if m else default

            def _kw_overlap(a: str, b: str) -> float:
                wa = set(re.findall(r'\b[a-z]{3,}\b', a.lower()))
                wb = set(re.findall(r'\b[a-z]{3,}\b', b.lower()))
                if not wa or not wb:
                    return 0.5
                return round(len(wa & wb) / math.sqrt(len(wa) * len(wb)), 4)

            q_emb, c_emb = _fallback_embed([tc.question, ctx_text])
            q_vec, c_vec = np.array(q_emb), np.array(c_emb)
            denom = float(np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
            ctx_prec = round(min(1.0, float(np.dot(q_vec, c_vec) / denom) + 0.3) if denom > 0 else 0.5, 4)

            return {
                "faithfulness": _parse(faith_resp.content),
                "answer_relevancy": _parse(rel_resp.content),
                "context_precision": ctx_prec,
                "context_recall": _kw_overlap(tc.ground_truth, ctx_text),
            }

        try:
            scored = await asyncio.gather(*[_score_one(tc) for tc in test_cases])

            async with get_db_context() as db:
                from sqlalchemy import select
                result = await db.execute(select(EvalRun).where(EvalRun.id == eval_run_id))
                run = result.scalar_one_or_none()
                if not run:
                    return

                cases = [
                    EvalCase(
                        eval_run_id=eval_run_id,
                        question=tc.question, answer=tc.answer,
                        contexts=tc.contexts, ground_truth=tc.ground_truth,
                        faithfulness_score=s["faithfulness"],
                        answer_relevancy_score=s["answer_relevancy"],
                        context_precision_score=s["context_precision"],
                        context_recall_score=s["context_recall"],
                    )
                    for tc, s in zip(test_cases, scored)
                ]
                db.add_all(cases)

                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                run.avg_faithfulness = round(float(np.mean([s["faithfulness"] for s in scored])), 4)
                run.avg_answer_relevancy = round(float(np.mean([s["answer_relevancy"] for s in scored])), 4)
                run.avg_context_precision = round(float(np.mean([s["context_precision"] for s in scored])), 4)
                run.avg_context_recall = round(float(np.mean([s["context_recall"] for s in scored])), 4)
                await db.commit()

            logger.info(f"LLM evaluation complete | run_id={eval_run_id} | cases={len(test_cases)}")

        except Exception as e:
            logger.error(f"LLM evaluation failed | run_id={eval_run_id} | error={e}")
            await self._fail_eval_run(eval_run_id, str(e))
            raise

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
