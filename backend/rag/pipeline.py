"""rag/pipeline.py - End-to-end RAG pipeline."""
import os
from typing import Optional
from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from backend.config import get_settings
from backend.rag.vectorstore import VectorStoreManager, get_vector_store
from backend.rag.retriever import RAGRetriever

settings = get_settings()

# Confidence thresholds calibrated for TF-IDF hash embeddings.
# TF-IDF cosine scores range roughly -0.5 to +0.5 (vs 0.3–0.95 for semantic).
# We never block retrieval based on score — only classify confidence level.
_CONF_HIGH   = 0.20   # top chunk score >= this → high
_CONF_MEDIUM = 0.0    # top chunk score >= this → medium (including slightly negative)


RAG_SYSTEM_PROMPT = """You are a trustworthy AI assistant that answers questions strictly from the provided context.

RULES:
1. Base your answer ONLY on the context passages below.
2. Cite the source in brackets after each key point, e.g. [source: attention-is-all-you-need.txt].
3. If the context contains a partial answer, give the partial answer and state what is missing.
4. Only say "I cannot find this in the provided documents" if the context is completely unrelated to the question.
5. Be concise and direct.

Context passages:
{context}"""

RAG_HUMAN_PROMPT = "Question: {question}"


def _build_context(chunks: list[dict]) -> tuple[str, float]:
    """
    Build context string from retrieved chunks.

    Returns (context_str, top_score).
    - Never filters chunks by score; the retriever already applies MMR ranking.
    - Includes ALL returned chunks so the LLM has maximum grounding.
    - top_score is used only for the confidence badge in the UI.
    """
    if not chunks:
        return "", 0.0

    parts = []
    for i, c in enumerate(chunks):
        score_pct = max(0, int(c.get("score", 0) * 100))
        parts.append(
            f"[Passage {i + 1} | source: {c['source']} | relevance: {score_pct}%]\n{c['content']}"
        )

    top_score = chunks[0].get("score", 0.0)
    return "\n\n---\n\n".join(parts), top_score


def _confidence_label(top_score: float, chunk_count: int) -> str:
    """
    Confidence label based on retrieval score and chunk count.

    Calibrated for TF-IDF hash embeddings (score range ~ -0.5 to +0.5).
    """
    if chunk_count == 0:
        return "low"
    if top_score >= _CONF_HIGH:
        return "high"
    if top_score >= _CONF_MEDIUM:
        return "medium"
    return "low"


class RAGPipeline:
    def __init__(self, vectorstore: Optional[VectorStoreManager] = None):
        self.vector_store = vectorstore or get_vector_store()
        self.retriever = RAGRetriever(vector_store=self.vector_store)

        if settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

        if settings.groq_api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.groq_temperature,
            )
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT),
            ])
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            self.llm = None
            self.prompt = None
            self.chain = None
            logger.warning("GROQ_API_KEY not set - RAG query will return degraded response")

    @traceable(name="rag_pipeline_query", run_type="chain")
    def query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        use_mmr: bool = True,
        fetch_k: int = 20,
        mmr_lambda: Optional[float] = None,
    ) -> dict:
        """
        Returns dict with keys:
          question, answer, retrieved_chunks, model_used, confidence
        """
        logger.info(
            f"RAG query | question={question[:80]} | "
            f"top_k={top_k or 'default'} | mmr={use_mmr}"
        )

        chunks = self.retriever.retrieve(
            query=question,
            collection_name=collection_name,
            top_k=top_k,
            use_mmr=use_mmr,
            fetch_k=fetch_k,
            mmr_lambda=mmr_lambda,
        )

        if not self.chain:
            return {
                "question": question,
                "answer": "RAG pipeline is not configured. Set GROQ_API_KEY in your environment variables.",
                "retrieved_chunks": [],
                "model_used": "not-configured",
                "confidence": "low",
            }

        if not chunks:
            logger.warning("No chunks retrieved from vector store")
            return {
                "question": question,
                "answer": "No documents are indexed yet. Please upload documents first, or ask about AI/ML topics using the pre-loaded documents.",
                "retrieved_chunks": [],
                "model_used": settings.groq_model,
                "confidence": "low",
            }

        context_str, top_score = _build_context(chunks)
        confidence = _confidence_label(top_score, len(chunks))

        answer = self.chain.invoke({"context": context_str, "question": question})

        logger.info(
            f"RAG response | chunks={len(chunks)} | top_score={top_score:.3f} | "
            f"confidence={confidence} | answer_len={len(answer)}"
        )
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "model_used": settings.groq_model,
            "confidence": confidence,
        }

    def ingest_text(
        self,
        text: str,
        filename: str,
        collection_name: Optional[str] = None,
    ) -> dict:
        return self.vector_store.ingest_text(
            text=text, filename=filename, collection_name=collection_name
        )

    def ingest_pdf(self, file_path: str, collection_name: Optional[str] = None) -> dict:
        return self.vector_store.ingest_pdf(
            file_path=file_path, collection_name=collection_name
        )


_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline()
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=f"RAG pipeline unavailable: {e}. Check GROQ_API_KEY.",
            ) from e
    return _pipeline
