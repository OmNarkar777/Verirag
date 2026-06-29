"""rag/pipeline.py - End-to-end RAG pipeline."""
import os
from typing import Optional
from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

from backend.config import get_settings
from backend.rag.vectorstore import VectorStoreManager, get_vector_store
from backend.rag.retriever import RAGRetriever

settings = get_settings()

# Minimum similarity score for a chunk to be included in context.
# Chunks below this threshold are too dissimilar to the query to be useful
# and risk grounding the answer in irrelevant content.
_MIN_CHUNK_SCORE = 0.10

RAG_SYSTEM_PROMPT = """You are a precise, trustworthy AI assistant. Your answers must be \
strictly grounded in the provided context passages.

Rules:
1. Answer only from the context below. Never use external or prior knowledge.
2. If the context is insufficient, say exactly: "I don't have enough context to answer this question."
3. When you answer, cite the source document name in brackets, e.g. [attention-is-all-you-need.txt].
4. Be concise and direct. Do not repeat the question.
5. If the question is partially answerable, answer what you can and note what is missing.

Context passages (ordered by relevance):
{context}"""

RAG_HUMAN_PROMPT = "Question: {question}"


def _build_context(chunks: list[dict]) -> tuple[str, float]:
    """
    Build the context string from retrieved chunks and compute a confidence score.

    Confidence is derived from the top chunk's similarity score:
    - >= 0.70  high   (strong semantic match)
    - >= 0.40  medium (partial match)
    - <  0.40  low    (weak match — answer may be unreliable)

    Returns (context_str, top_score).
    """
    filtered = [c for c in chunks if c.get("score", 0) >= _MIN_CHUNK_SCORE]
    if not filtered:
        return "", 0.0

    parts = []
    for i, c in enumerate(filtered):
        score_pct = int(c["score"] * 100)
        parts.append(
            f"[Passage {i+1} | source: {c['source']} | relevance: {score_pct}%]\n{c['content']}"
        )

    return "\n\n---\n\n".join(parts), filtered[0]["score"]


def _confidence_label(top_score: float) -> str:
    if top_score >= 0.70:
        return "high"
    if top_score >= 0.40:
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
    ) -> dict:
        """
        Returns dict with keys:
          question, answer, retrieved_chunks, model_used, confidence
        """
        logger.info(f"RAG query | question={question[:80]}")

        chunks = self.retriever.retrieve(
            query=question,
            collection_name=collection_name,
            top_k=top_k,
            use_mmr=True,
        )

        if not self.chain:
            return {
                "question": question,
                "answer": "RAG pipeline is not configured. Set GROQ_API_KEY in your environment variables.",
                "retrieved_chunks": [],
                "model_used": "not-configured",
                "confidence": "low",
            }

        context_str, top_score = _build_context(chunks)

        if not context_str:
            logger.warning("No usable chunks retrieved (all below threshold or empty)")
            return {
                "question": question,
                "answer": "I don't have enough context to answer this question. Try uploading relevant documents first.",
                "retrieved_chunks": chunks,
                "model_used": settings.groq_model,
                "confidence": "low",
            }

        confidence = _confidence_label(top_score)
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
