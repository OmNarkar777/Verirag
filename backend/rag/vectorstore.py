"""
rag/vectorstore.py — Dual-mode vector store: ChromaDB (local) or PostgreSQL+numpy (Vercel).

EMBEDDING STRATEGY:
  HuggingFace Inference API → sentence-transformers/all-MiniLM-L6-v2
  384-dimensional embeddings, cosine similarity space.
  No local model weights — keeps the bundle small and deployment simple.

BACKEND SELECTION:
  ChromaDB  → used when importable (local dev, Docker).
  PostgresVectorStore → used on Vercel (ChromaDB requires libgomp.so.1 which is
    absent in Vercel Lambda). Stores JSONB float arrays in PostgreSQL, performs
    cosine similarity with numpy. Works identically from the caller's perspective.

CHUNKING STRATEGY:
  chunk_size=512, chunk_overlap=50, semantic separator order.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment]
    _CHROMADB_AVAILABLE = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from loguru import logger

from backend.config import get_settings

settings = get_settings()

_EMBED_DIM = 384


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _fallback_embed(texts: list[str], dim: int = _EMBED_DIM) -> list[list[float]]:
    """
    Pure-Python TF-IDF hash embedding. No external dependencies.

    Used when HuggingFace Inference API is unavailable (no HF_TOKEN, rate limit,
    or network error). Provides keyword-based semantic similarity: texts sharing
    important terms are placed closer in embedding space.

    Sufficient for a demonstration RAG pipeline where the recruiter queries
    with domain terms that appear in the indexed documents.
    """
    import re
    import math
    import numpy as np

    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    embeddings = []
    for text in texts:
        words = tokenize(text)
        if not words:
            embeddings.append([0.0] * dim)
            continue
        tf: dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        vec = np.zeros(dim, dtype=np.float64)
        for word, count in tf.items():
            idf = 1.0 / (1.0 + math.log1p(count))
            for seed in range(4):
                idx = abs(hash(f"{word}_{seed}")) % dim
                sign = 1 if abs(hash(f"{word}_{seed}_s")) % 2 else -1
                vec[idx] += sign * count * idf
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec.tolist())
    return embeddings


def _embed_texts(hf_client: InferenceClient, texts: list[str]) -> list[list[float]]:
    """
    Embed texts. Uses local TF-IDF fallback when HF_TOKEN is absent (default on Vercel).
    Only calls api-inference.huggingface.co when a token is explicitly configured.
    """
    if not settings.hf_token:
        return _fallback_embed(texts)

    try:
        response = hf_client.feature_extraction(
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
        logger.warning(f"HF Inference API error ({str(e)[:80]}); using local fallback embedder")
        return _fallback_embed(texts)


# ── PostgreSQL-backed vector store ────────────────────────────────────────────

class PostgresVectorStore:
    """
    PostgreSQL-backed vector store: JSONB embeddings + numpy cosine similarity.

    No native libraries required — works on Vercel Lambda and any Python env.
    Suitable for demo-scale workloads (< 10 000 chunks, < 5 concurrent queries).

    Embedding matrix is cached in memory per collection to avoid redundant DB
    round-trips within a single Lambda invocation.
    """

    def __init__(self):
        self._hf_client = InferenceClient(token=settings.hf_token or None)
        self._splitter = _build_text_splitter()
        # In-memory cache: collection → list of (id, doc_id, content, embedding, metadata)
        self._cache: dict[str, list[dict]] = {}
        embed_mode = f"HF Inference API ({settings.embedding_model})" if settings.hf_token else "local TF-IDF fallback"
        logger.info(f"PostgresVectorStore initialized | embedding={embed_mode}")

    # ── Internal DB helpers ───────────────────────────────────────────────────

    def _get_engine(self):
        from backend.database import _get_sync_engine
        return _get_sync_engine()

    def _session(self):
        from sqlalchemy.orm import sessionmaker
        factory = sessionmaker(bind=self._get_engine(), expire_on_commit=False)
        return factory()

    def _invalidate_cache(self, collection_name: str) -> None:
        self._cache.pop(collection_name, None)

    def _load_collection(self, collection_name: str) -> list[dict]:
        """Load all chunks for a collection from PostgreSQL (cached per invocation)."""
        if collection_name in self._cache:
            return self._cache[collection_name]

        from sqlalchemy import select
        from backend.models import DocumentChunk

        with self._session() as sess:
            rows = sess.execute(
                select(DocumentChunk)
                .where(DocumentChunk.collection_name == collection_name)
                .order_by(DocumentChunk.doc_id, DocumentChunk.chunk_index)
            ).scalars().all()

        chunks = [
            {
                "id": str(row.id),
                "doc_id": row.doc_id,
                "content": row.content,
                "embedding": row.embedding,
                "source": row.filename,
                "metadata": row.chunk_metadata,
            }
            for row in rows
        ]
        self._cache[collection_name] = chunks
        return chunks

    def _store_chunks(
        self,
        collection_name: str,
        doc_id: str,
        filename: str,
        chunk_texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Upsert chunks into document_chunks table (delete-then-insert for idempotency)."""
        from sqlalchemy import delete
        from backend.models import DocumentChunk

        with self._session() as sess:
            sess.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.doc_id == doc_id,
                    DocumentChunk.collection_name == collection_name,
                )
            )
            for i, (text, emb, meta) in enumerate(zip(chunk_texts, embeddings, metadatas)):
                sess.add(DocumentChunk(
                    doc_id=doc_id,
                    filename=filename,
                    chunk_index=i,
                    content=text,
                    embedding=emb,
                    collection_name=collection_name,
                    chunk_metadata=meta,
                ))
            sess.commit()

        self._invalidate_cache(collection_name)

    # ── Core ingestion ────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        filename: str,
        collection_name: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        collection_name = collection_name or settings.chroma_collection_name
        doc_id = hashlib.sha256(f"{filename}:{text[:100]}".encode()).hexdigest()[:16]

        chunks = _build_text_splitter().create_documents(
            texts=[text],
            metadatas=[{"source": filename, "doc_id": doc_id}],
        )
        if not chunks:
            raise ValueError(f"No chunks produced from document: {filename}")

        chunk_texts = [c.page_content for c in chunks]
        metadatas = [
            {"source": filename, "doc_id": doc_id, "chunk_index": i,
             "total_chunks": len(chunks), **(extra_metadata or {})}
            for i in range(len(chunks))
        ]
        embeddings = _embed_texts(self._hf_client, chunk_texts)

        self._store_chunks(collection_name, doc_id, filename, chunk_texts, embeddings, metadatas)

        logger.info(f"Ingested | filename={filename} | chunks={len(chunks)} | collection={collection_name}")
        return {"doc_id": doc_id, "filename": filename, "chunks_created": len(chunks), "collection_name": collection_name}

    def ingest_pdf(self, file_path: str, collection_name: Optional[str] = None) -> dict:
        from langchain_community.document_loaders import PyPDFLoader
        path = Path(file_path)
        pages = PyPDFLoader(str(path)).load()
        full_text = "\n\n".join(p.page_content for p in pages)
        return self.ingest_text(
            text=full_text, filename=path.name, collection_name=collection_name,
            extra_metadata={"page_count": len(pages), "file_type": "pdf"},
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        import numpy as np
        collection_name = collection_name or settings.chroma_collection_name
        top_k = top_k or settings.retrieval_top_k
        chunks = self._load_collection(collection_name)
        if not chunks:
            return []

        q_emb = np.array(_embed_texts(self._hf_client, [query])[0])
        embs = np.array([c["embedding"] for c in chunks])

        # Cosine similarity
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-10
        scores = (embs @ q_emb) / norms

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"content": chunks[i]["content"], "source": chunks[i]["source"],
             "score": float(scores[i]), "metadata": chunks[i]["metadata"]}
            for i in top_idx
        ]

    def mmr_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: Optional[float] = None,
    ) -> list[dict]:
        import numpy as np
        collection_name = collection_name or settings.chroma_collection_name
        top_k = top_k or settings.retrieval_top_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.retrieval_lambda
        chunks = self._load_collection(collection_name)
        if not chunks:
            return []

        q_emb = np.array(_embed_texts(self._hf_client, [query])[0])
        embs = np.array([c["embedding"] for c in chunks])

        # Relevance scores
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-10
        relevance = (embs @ q_emb) / norms

        # Fetch top candidates
        n_candidates = min(fetch_k, len(chunks))
        candidate_idx = list(np.argsort(relevance)[::-1][:n_candidates])

        selected: list[int] = []
        remaining = candidate_idx[:]

        for _ in range(min(top_k, n_candidates)):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_embs = embs[selected]
                best = max(
                    remaining,
                    key=lambda i: (
                        lambda_mult * relevance[i]
                        - (1 - lambda_mult) * float(np.max(
                            (embs[i] @ sel_embs.T) /
                            (np.linalg.norm(embs[i]) * np.linalg.norm(sel_embs, axis=1) + 1e-10)
                        ))
                    ),
                )
            selected.append(best)
            remaining.remove(best)

        return [
            {"content": chunks[i]["content"], "source": chunks[i]["source"],
             "score": float(relevance[i]), "metadata": chunks[i]["metadata"]}
            for i in selected
        ]

    def get_collection_stats(self, collection_name: Optional[str] = None) -> dict:
        from sqlalchemy import select, func
        from backend.models import DocumentChunk
        collection_name = collection_name or settings.chroma_collection_name
        with self._session() as sess:
            count = sess.execute(
                select(func.count(DocumentChunk.id))
                .where(DocumentChunk.collection_name == collection_name)
            ).scalar_one()
        return {"collection_name": collection_name, "document_count": count}

    def doc_exists(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        from sqlalchemy import select, func
        from backend.models import DocumentChunk
        collection_name = collection_name or settings.chroma_collection_name
        with self._session() as sess:
            count = sess.execute(
                select(func.count(DocumentChunk.id))
                .where(DocumentChunk.doc_id == doc_id,
                       DocumentChunk.collection_name == collection_name)
            ).scalar_one()
        return count > 0


# ── ChromaDB-backed vector store (local dev) ─────────────────────────────────

class VectorStoreManager:
    """
    ChromaDB-backed vector store for local development / Docker.

    Falls back automatically to PostgresVectorStore on Vercel where
    chromadb is unavailable (libgomp.so.1 missing in Lambda runtime).
    """

    def __init__(self):
        if not _CHROMADB_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. "
                "On Vercel, PostgresVectorStore is used instead. "
                "For local full-RAG development: pip install chromadb==0.5.0"
            )
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._hf_client = InferenceClient(token=settings.hf_token or None)
        self._splitter = _build_text_splitter()
        logger.info(
            f"VectorStoreManager (ChromaDB) initialized | "
            f"embedding_model={settings.embedding_model} | "
            f"persist_dir={settings.chroma_persist_dir}"
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return _embed_texts(self._hf_client, texts)

    def get_or_create_collection(self, collection_name: str):
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_text(self, text: str, filename: str, collection_name=None, extra_metadata=None) -> dict:
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        doc_id = hashlib.sha256(f"{filename}:{text[:100]}".encode()).hexdigest()[:16]

        chunks = self._splitter.create_documents(
            texts=[text], metadatas=[{"source": filename, "doc_id": doc_id}],
        )
        if not chunks:
            raise ValueError(f"No chunks produced from document: {filename}")

        chunk_texts = [c.page_content for c in chunks]
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadata = [
            {"source": filename, "doc_id": doc_id, "chunk_index": i,
             "total_chunks": len(chunks), **(extra_metadata or {})}
            for i in range(len(chunks))
        ]
        embeddings = self._embed(chunk_texts)
        collection.upsert(ids=chunk_ids, documents=chunk_texts, embeddings=embeddings, metadatas=chunk_metadata)

        logger.info(f"Ingested | filename={filename} | chunks={len(chunks)} | collection={collection_name}")
        return {"doc_id": doc_id, "filename": filename, "chunks_created": len(chunks), "collection_name": collection_name}

    def ingest_pdf(self, file_path: str, collection_name=None) -> dict:
        from langchain_community.document_loaders import PyPDFLoader
        path = Path(file_path)
        pages = PyPDFLoader(str(path)).load()
        full_text = "\n\n".join(p.page_content for p in pages)
        return self.ingest_text(text=full_text, filename=path.name, collection_name=collection_name,
                                extra_metadata={"page_count": len(pages), "file_type": "pdf"})

    def similarity_search(self, query: str, collection_name=None, top_k=None) -> list[dict]:
        collection_name = collection_name or settings.chroma_collection_name
        top_k = top_k or settings.retrieval_top_k
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []
        q_emb = self._embed([query])[0]
        results = collection.query(
            query_embeddings=[q_emb], n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            {"content": doc, "source": meta.get("source", "unknown"),
             "score": 1.0 - dist, "metadata": meta}
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ]

    def mmr_search(self, query: str, collection_name=None, top_k=None, fetch_k=20, lambda_mult=None) -> list[dict]:
        collection_name = collection_name or settings.chroma_collection_name
        top_k = top_k or settings.retrieval_top_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.retrieval_lambda
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []

        import numpy as np
        q_emb = self._embed([query])[0]
        candidates = collection.query(
            query_embeddings=[q_emb], n_results=min(fetch_k, collection.count()),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if not candidates["documents"][0]:
            return []

        selected_indices = self._mmr_select(
            query_embedding=q_emb, candidate_embeddings=candidates["embeddings"][0],
            top_k=min(top_k, len(candidates["documents"][0])), lambda_mult=lambda_mult,
        )
        return [
            {"content": candidates["documents"][0][i], "source": candidates["metadatas"][0][i].get("source", "unknown"),
             "score": 1.0 - candidates["distances"][0][i], "metadata": candidates["metadatas"][0][i]}
            for i in selected_indices
        ]

    @staticmethod
    def _mmr_select(query_embedding, candidate_embeddings, top_k, lambda_mult) -> list[int]:
        import numpy as np
        q_vec = np.array(query_embedding)
        c_vecs = np.array(candidate_embeddings)
        relevance = (c_vecs @ q_vec) / (np.linalg.norm(c_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-10)
        selected: list[int] = []
        remaining = list(range(len(candidate_embeddings)))
        for _ in range(top_k):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_vecs = c_vecs[selected]
                best = max(remaining, key=lambda i: (
                    lambda_mult * relevance[i] - (1 - lambda_mult) * float(np.max(
                        (c_vecs[i] @ sel_vecs.T) / (np.linalg.norm(c_vecs[i]) * np.linalg.norm(sel_vecs, axis=1) + 1e-10)
                    ))
                ))
            selected.append(best)
            remaining.remove(best)
        return selected

    def get_collection_stats(self, collection_name=None) -> dict:
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        return {"collection_name": collection_name, "document_count": collection.count()}

    def doc_exists(self, doc_id: str, collection_name=None) -> bool:
        collection_name = collection_name or settings.chroma_collection_name
        collection = self.get_or_create_collection(collection_name)
        try:
            results = collection.get(ids=[f"{doc_id}_chunk_0"])
            return bool(results["ids"])
        except Exception:
            return False


# ── Factory — auto-selects backend ───────────────────────────────────────────

_vector_store: VectorStoreManager | PostgresVectorStore | None = None


def get_vector_store() -> VectorStoreManager | PostgresVectorStore:
    """
    Return the appropriate vector store:
      - ChromaDB (VectorStoreManager) when chromadb is importable (local dev)
      - PostgresVectorStore when chromadb is unavailable (Vercel)
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if _CHROMADB_AVAILABLE:
        try:
            _vector_store = VectorStoreManager()
            return _vector_store
        except Exception as e:
            logger.warning(f"ChromaDB init failed, falling back to PostgresVectorStore: {e}")

    _vector_store = PostgresVectorStore()
    return _vector_store
