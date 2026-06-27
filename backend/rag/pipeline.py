"""rag/pipeline.py â€” End-to-end RAG pipeline with correct interface for routers."""
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

RAG_SYSTEM_PROMPT = """You are a precise assistant. Answer ONLY from the provided context.
If the context lacks the answer, say so clearly. Do not use external knowledge.

Context:
{context}"""

RAG_HUMAN_PROMPT = "Question: {question}"


class RAGPipeline:
    def __init__(self, vectorstore: Optional[VectorStoreManager] = None):
        # Use 'vector_store' attribute â€” routers reference pipeline.vector_store
        self.vector_store = vectorstore or get_vector_store()
        self.retriever = RAGRetriever(vector_store=self.vector_store)

        # LangSmith env setup inside __init__ â€” avoids module-level import failures
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
            logger.warning("GROQ_API_KEY not set — RAG query will return degraded response")

    @traceable(name="rag_pipeline_query", run_type="chain")
    def query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Returns dict with keys:
          question, answer, retrieved_chunks (list of dicts), model_used
        These keys match exactly what routers/pipeline.py expects.
        """
        logger.info(f"RAG query | question={question[:80]}")

        # Use keyword args â€” avoids positional-arg bug (top_k passed as collection_name)
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
                "retrieved_chunks": chunks if chunks else [],
                "model_used": "not-configured",
            }

        if not chunks:
            logger.warning("No chunks retrieved")
            return {
                "question": question,
                "answer": "I don't have enough context to answer this question.",
                "retrieved_chunks": [],
                "model_used": settings.groq_model,
            }

        context_str = "\n\n".join(
            f"[Chunk {i+1} from {c['source']}]:\n{c['content']}"
            for i, c in enumerate(chunks)
        )

        answer = self.chain.invoke({"context": context_str, "question": question})

        logger.info(f"RAG response | chunks={len(chunks)} | answer_len={len(answer)}")
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,      # list of {content, source, score, metadata}
            "model_used": settings.groq_model,
        }

    def ingest_text(
        self,
        text: str,
        filename: str,
        collection_name: Optional[str] = None,
    ) -> dict:
        """Delegate to vectorstore â€” accepts collection_name for router compatibility."""
        return self.vector_store.ingest_text(
            text=text, filename=filename, collection_name=collection_name
        )

    def ingest_pdf(self, file_path: str, collection_name: Optional[str] = None) -> dict:
        """Delegate to vectorstore."""
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
                detail=f"RAG pipeline unavailable: {e}. Check GROQ_API_KEY and HF_TOKEN.",
            ) from e
    return _pipeline