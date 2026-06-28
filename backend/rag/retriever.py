"""
rag/retriever.py — Retrieval logic with LangSmith tracing.

WHY SEPARATE RETRIEVER FROM VECTORSTORE:
- vectorstore.py handles storage mechanics (ChromaDB operations)
- retriever.py handles retrieval strategy (which algorithm, how many docs, post-filtering)
- This separation lets us swap retrieval strategies (BM25, hybrid, reranking)
  without touching storage code — a common RAG architecture pattern
"""
from __future__ import annotations

from langsmith import traceable
from loguru import logger

from backend.config import get_settings
from backend.rag.vectorstore import VectorStoreManager, get_vector_store

settings = get_settings()


class RAGRetriever:
    """
    Retrieval layer over ChromaDB with LangSmith tracing.
    
    @traceable decorator: every retrieve() call creates a LangSmith trace span.
    This is invaluable for debugging — you can see exactly which chunks were
    retrieved for each question that gets a bad RAGAS score.
    """

    def __init__(self, vector_store: VectorStoreManager | None = None):
        self.vector_store = vector_store or get_vector_store()

    @traceable(name="rag_retrieval", run_type="retriever")
    def retrieve(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        use_mmr: bool = True,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        
        Returns list of {content, source, score, metadata} dicts.
        
        WHY MMR BY DEFAULT:
        In RAGAS eval, context_precision measures whether the retrieved chunks
        are USEFUL for generating the correct answer. With vanilla similarity
        search, if the top-5 chunks are near-duplicates, the LLM gets
        redundant info — hurting both faithfulness and context_precision.
        MMR diversifies the retrieved set, giving the LLM more signal.
        """
        collection_name = collection_name or settings.chroma_collection_name
        top_k = top_k or settings.retrieval_top_k

        logger.debug(
            f"Retrieving chunks | query_len={len(query)} | "
            f"top_k={top_k} | mmr={use_mmr} | collection={collection_name}"
        )

        if use_mmr:
            chunks = self.vector_store.mmr_search(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
            )
        else:
            chunks = self.vector_store.similarity_search(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
            )

        top_score = f"{chunks[0]['score']:.3f}" if chunks else "0"
        logger.debug(f"Retrieved {len(chunks)} chunks | top_score={top_score}")
        return chunks

    def retrieve_for_ragas(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        """
        Convenience method that returns just the text content strings.
        RAGAS expects contexts as list[str], not list[dict].
        """
        chunks = self.retrieve(query=query, collection_name=collection_name, top_k=top_k)
        return [chunk["content"] for chunk in chunks]

