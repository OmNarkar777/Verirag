"""
rag/vectorstore.py — ChromaDB setup and document ingestion.

EMBEDDING STRATEGY:
Using sentence-transformers locally (all-MiniLM-L6-v2) instead of OpenAI embeddings:
- No API cost per document (critical for eval pipelines that re-ingest often)
- No rate limiting during bulk ingestion
- Reproducible: same model = same embeddings every time (eval consistency)
- 384 dimensions: fast similarity search, low memory footprint
- Quality: good enough for domain-general RAG; upgrade to BAAI/bge-large for prod

CHUNKING STRATEGY:
- chunk_size=512 tokens: balances context richness vs retrieval precision
  Too small (< 128): loses context, poor answers
  Too large (> 1024): retrieves irrelevant content, hurts faithfulness score
- chunk_overlap=50: prevents information loss at chunk boundaries
"""

import hashlib
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from huggingface_hub import InferenceClient
from loguru import logger

from backend.config import get_settings

settings = get_settings()


class VectorStoreManager:
    """
    Wraps ChromaDB operations with a clean interface.

    WHY NOT langchain's ChromaDB wrapper:
    Using ChromaDB's native client gives us more control over:
    - Custom embedding functions (our HF Inference API wrapper)
    - Batch operations with progress tracking
    - Collection management (list, delete, inspect)
    """

    def __init__(self):
        # PersistentClient: data survives process restarts
        # EphemeralClient: in-memory only (good for testing)
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._hf_client = InferenceClient(token=settings.hf_token or None)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            # Split on semantic boundaries first, then fall back to chars
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"VectorStoreManager initialized | "
            f"embedding_model={settings.embedding_model} (HF Inference API) | "
            f"persist_dir={settings.chroma_persist_dir}"
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via HuggingFace Inference API (no local model weights needed)."""
        response = self._hf_client.feature_extraction(
            text=texts,
            model=settings.embedding_model,
            normalize=True,
        )
        # response shape: (n_texts, dim) as nested list
        if hasattr(response, "tolist"):
            return response.tolist()
        return [list(vec) for vec in response]

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get existing collection or create new one.
        cosine distance: standard for semantic similarity search.
        """
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_text(
        self,
        text: str,
        filename: str,
        collection_name: str | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        """
        Ingest raw text into ChromaDB.
        
        Returns ingestion stats for API response and PostgreSQL tracking.
        """
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)

        # Deterministic doc_id: same file = same ID prevents duplicates
        doc_id = hashlib.sha256(f"{filename}:{text[:100]}".encode()).hexdigest()[:16]

        chunks = self._text_splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename, "doc_id": doc_id}],
        )

        if not chunks:
            raise ValueError(f"No chunks produced from document: {filename}")

        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadata = [
            {
                "source": filename,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(extra_metadata or {}),
            }
            for i in range(len(chunks))
        ]

        embeddings = self._embed(chunk_texts)

        # upsert: idempotent — re-ingesting the same doc updates chunks, not duplicates
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadata,
        )

        logger.info(
            f"Ingested document | filename={filename} | "
            f"chunks={len(chunks)} | collection={collection_name}"
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "collection_name": collection_name,
        }

    def ingest_pdf(self, file_path: str, collection_name: str | None = None) -> dict:
        """Ingest a PDF file by extracting text and delegating to ingest_text."""
        path = Path(file_path)
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        full_text = "\n\n".join(page.page_content for page in pages)
        return self.ingest_text(
            text=full_text,
            filename=path.name,
            collection_name=collection_name,
            extra_metadata={"page_count": len(pages), "file_type": "pdf"},
        )

    def similarity_search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Standard cosine similarity search.
        Returns list of {content, source, score, metadata}.
        """
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        top_k = top_k or settings.retrieval_top_k

        if collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        return [
            {
                "content": doc,
                "source": meta.get("source", "unknown"),
                # ChromaDB returns distance (lower=closer); convert to similarity score
                "score": 1.0 - dist,
                "metadata": meta,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def mmr_search(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float | None = None,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance (MMR) search.
        
        WHY MMR:
        Standard similarity search returns the most similar chunks — but they
        might all be near-duplicates (e.g., same paragraph repeated). MMR
        penalizes redundancy, returning a DIVERSE set of relevant chunks.
        
        This matters for RAGAS context_precision: diverse chunks give the LLM
        more informational surface area, reducing hallucination risk.
        
        lambda_mult: 0.0 = maximize diversity, 1.0 = maximize similarity
        0.5 is a good default (per MMR paper, Carbonell & Goldstein 1998)
        """
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        top_k = top_k or settings.retrieval_top_k
        lambda_mult = lambda_mult or settings.retrieval_lambda

        if collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]

        # Fetch more candidates than needed, then MMR-select the final top_k
        candidates = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, collection.count()),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        if not candidates["documents"][0]:
            return []

        # Run MMR selection algorithm
        selected_indices = self._mmr_select(
            query_embedding=query_embedding,
            candidate_embeddings=candidates["embeddings"][0],
            top_k=min(top_k, len(candidates["documents"][0])),
            lambda_mult=lambda_mult,
        )

        return [
            {
                "content": candidates["documents"][0][i],
                "source": candidates["metadatas"][0][i].get("source", "unknown"),
                "score": 1.0 - candidates["distances"][0][i],
                "metadata": candidates["metadatas"][0][i],
            }
            for i in selected_indices
        ]

    @staticmethod
    def _mmr_select(
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        top_k: int,
        lambda_mult: float,
    ) -> list[int]:
        """
        Pure MMR selection over pre-fetched candidates.
        
        Score = λ * relevance(query, doc) - (1-λ) * max_similarity(doc, selected)
        
        Iteratively selects the candidate that maximizes this score,
        balancing query relevance against redundancy with already-selected docs.
        """
        import numpy as np

        query_vec = np.array(query_embedding)
        cand_vecs = np.array(candidate_embeddings)

        # Relevance scores: cosine similarity to query
        relevance = (cand_vecs @ query_vec) / (
            np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )

        selected: list[int] = []
        remaining = list(range(len(candidate_embeddings)))

        for _ in range(top_k):
            if not remaining:
                break

            if not selected:
                # First selection: pure relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # Subsequent: MMR score
                selected_vecs = cand_vecs[selected]
                best = max(
                    remaining,
                    key=lambda i: (
                        lambda_mult * relevance[i]
                        - (1 - lambda_mult)
                        * float(
                            np.max(
                                (cand_vecs[i] @ selected_vecs.T)
                                / (
                                    np.linalg.norm(cand_vecs[i])
                                    * np.linalg.norm(selected_vecs, axis=1)
                                    + 1e-10
                                )
                            )
                        )
                    ),
                )

            selected.append(best)
            remaining.remove(best)

        return selected

    def get_collection_stats(self, collection_name: str | None = None) -> dict:
        """Returns count and sample of documents in collection."""
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        return {
            "collection_name": collection_name,
            "document_count": collection.count(),
        }


# Module-level singleton — shared across the app lifecycle
# Initialized once in main.py lifespan, avoids model reload overhead
_vector_store: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    """FastAPI dependency for VectorStoreManager."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store
