"""
tests/test_pipeline.py - Tests for RAG pipeline components.

TESTING APPROACH:
- VectorStore: use real ChromaDB with ephemeral (in-memory) storage
- HF embeddings: patched to return fake vectors — avoids real API calls in CI
- RAG Pipeline: mock Groq API calls (we test retrieval logic, not Groq)
- Endpoints: mock pipeline and DB dependencies

WHAT WE VALIDATE:
1. Document ingestion creates correct chunk count
2. Similarity search returns relevant results
3. MMR search returns diverse results
4. Pipeline query returns correct structure
5. Endpoint responses have correct shape
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fake Embedding Helper ──────────────────────────────────────────────────────

def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake 4-dim embeddings — no HF API call."""
    return [[hash(t) % 100 / 100, 0.2, 0.3, 0.4] for t in texts]


# ── VectorStore Unit Tests ────────────────────────────────────────────────────

class TestVectorStoreManager:
    """
    Tests for ChromaDB operations.
    Uses PersistentClient with tmp_path for isolation.
    HF embeddings are patched to avoid real API calls.
    """

    @pytest.fixture
    def ephemeral_vs(self, tmp_path):
        """
        VectorStoreManager with temporary storage and fake embeddings.
        Each test gets a fresh ChromaDB instance.
        """
        with patch("backend.rag.vectorstore.settings") as mock_settings:
            mock_settings.chroma_persist_dir = str(tmp_path / "chroma")
            mock_settings.chroma_collection_name = "test_collection"
            mock_settings.embedding_model = "BAAI/bge-small-en-v1.5"
            mock_settings.retrieval_top_k = 3
            mock_settings.retrieval_lambda = 0.5
            mock_settings.hf_token = ""
            mock_settings.chunk_size = 200
            mock_settings.chunk_overlap = 20

            from backend.rag.vectorstore import VectorStoreManager
            vs = VectorStoreManager()
            # Patch _embed at instance level — bypasses all HF API calls
            vs._embed = _fake_embed
            yield vs

    def test_ingest_text_returns_metadata(self, ephemeral_vs):
        """Ingesting text should return doc_id, filename, and chunk count."""
        text = (
            "Transformers are a type of neural network architecture that uses self-attention. "
            "They were introduced in the paper Attention Is All You Need in 2017. "
            "The key innovation is the ability to process sequences in parallel. "
        ) * 5  # repeat to ensure multiple chunks

        result = ephemeral_vs.ingest_text(
            text=text,
            filename="test_doc.txt",
            collection_name="test_collection",
        )

        assert "doc_id" in result
        assert result["filename"] == "test_doc.txt"
        assert result["chunks_created"] >= 1
        assert result["collection_name"] == "test_collection"

    def test_ingest_produces_deterministic_doc_id(self, ephemeral_vs):
        """Same text + filename should produce the same doc_id (idempotency)."""
        text = "Test document content for determinism check."

        result1 = ephemeral_vs.ingest_text(text=text, filename="same.txt")
        result2 = ephemeral_vs.ingest_text(text=text, filename="same.txt")

        assert result1["doc_id"] == result2["doc_id"]

    def test_similarity_search_returns_results(self, ephemeral_vs):
        """After ingestion, similarity search should return relevant chunks."""
        ephemeral_vs.ingest_text(
            text="Transformers use self-attention mechanisms for NLP tasks.",
            filename="transformers.txt",
        )
        ephemeral_vs.ingest_text(
            text="ChromaDB is a vector database for semantic search.",
            filename="chromadb.txt",
        )

        results = ephemeral_vs.similarity_search(
            query="What are transformers in deep learning?",
            top_k=2,
        )

        assert len(results) > 0
        assert "content" in results[0]
        assert "source" in results[0]
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_similarity_search_empty_collection(self, ephemeral_vs):
        """Searching empty collection should return empty list."""
        results = ephemeral_vs.similarity_search(
            query="anything",
            collection_name="empty_collection",
            top_k=5,
        )
        assert results == []

    def test_mmr_search_returns_results(self, ephemeral_vs):
        """MMR search should return results after ingestion."""
        for i in range(3):
            ephemeral_vs.ingest_text(
                text=f"Transformers use self-attention. This is version {i}.",
                filename=f"dup_{i}.txt",
            )
        ephemeral_vs.ingest_text(
            text="ChromaDB stores vector embeddings for fast retrieval.",
            filename="different.txt",
        )

        mmr_results = ephemeral_vs.mmr_search(query="transformers attention", top_k=2)
        assert len(mmr_results) > 0

    def test_get_collection_stats(self, ephemeral_vs):
        """Collection stats should report correct document count after ingest."""
        initial_stats = ephemeral_vs.get_collection_stats()
        initial_count = initial_stats["document_count"]

        ephemeral_vs.ingest_text(
            text="A new document for stats testing. " * 10,
            filename="stats_test.txt",
        )

        updated_stats = ephemeral_vs.get_collection_stats()
        assert updated_stats["document_count"] > initial_count

    def test_mmr_select_algorithm(self):
        """MMR selection algorithm should prefer diverse candidates."""
        from backend.rag.vectorstore import VectorStoreManager

        query = [1.0, 0.0, 0.0]
        candidates = [
            [0.95, 0.05, 0.0],   # very similar to query
            [0.93, 0.07, 0.0],   # very similar to query AND candidate 0
            [0.90, 0.10, 0.0],   # similar but slightly more diverse
            [0.5, 0.5, 0.7],     # different direction — should be selected by MMR
        ]

        # With lambda=0 (max diversity), the diverse candidate should be prioritised
        selected = VectorStoreManager._mmr_select(
            query_embedding=query,
            candidate_embeddings=candidates,
            top_k=2,
            lambda_mult=0.0,
        )

        assert len(selected) == 2
        assert 3 in selected  # diverse candidate should be included


# ── Retriever Tests ───────────────────────────────────────────────────────────

class TestRAGRetriever:
    def test_retrieve_for_ragas_returns_strings(self):
        """retrieve_for_ragas should return list of strings, not dicts."""
        mock_store = MagicMock()
        mock_store.mmr_search.return_value = [
            {"content": "chunk 1", "source": "doc.txt", "score": 0.9, "metadata": {}},
            {"content": "chunk 2", "source": "doc.txt", "score": 0.8, "metadata": {}},
        ]

        from backend.rag.retriever import RAGRetriever

        retriever = RAGRetriever(vector_store=mock_store)
        results = retriever.retrieve_for_ragas(query="test query")

        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)
        assert results == ["chunk 1", "chunk 2"]


# ── Pipeline Endpoint Tests ───────────────────────────────────────────────────

def _make_mock_db():
    """Return an async generator mock for get_db."""
    from backend.database import get_db
    from backend.main import app

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    async def _mock_get_db():
        yield session

    return get_db, _mock_get_db, session


class TestPipelineEndpoints:
    """
    Endpoint tests use app.dependency_overrides — the correct way to mock FastAPI
    deps. patch() on a module attribute doesn't work because Depends() captures the
    function reference at definition time.
    """

    @pytest.fixture(autouse=True)
    def _setup_pipeline_mock(self):
        from backend.main import app
        from backend.rag.pipeline import get_pipeline

        self.mock_pipeline = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: self.mock_pipeline
        yield
        app.dependency_overrides.pop(get_pipeline, None)

    @pytest.mark.asyncio
    async def test_query_endpoint_structure(self):
        """POST /pipeline/query should return correct response structure."""
        from httpx import AsyncClient, ASGITransport
        from backend.main import app

        self.mock_pipeline.query.return_value = {
            "question": "What is RAG?",
            "answer": "RAG combines retrieval with generation.",
            "retrieved_chunks": [
                {
                    "content": "RAG is retrieval-augmented generation.",
                    "source": "rag_doc.txt",
                    "score": 0.92,
                    "metadata": {"chunk_index": 0},
                }
            ],
            "model_used": "llama-3.3-70b-versatile",
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/pipeline/query",
                json={"question": "What is RAG?", "top_k": 3},
            )

        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "retrieved_chunks" in data
        assert "model_used" in data
        assert len(data["retrieved_chunks"]) == 1
        assert data["retrieved_chunks"][0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_ingest_text_endpoint(self):
        """POST /pipeline/ingest/text should return 201 with chunk count."""
        from httpx import AsyncClient, ASGITransport
        from backend.main import app
        from backend.database import get_db

        self.mock_pipeline.ingest_text.return_value = {
            "doc_id": "abc123",
            "filename": "test.txt",
            "chunks_created": 3,
            "collection_name": "verirag_docs",
        }

        # ingest/text also Depends(get_db) for document tracking
        _, mock_get_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = mock_get_db
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/pipeline/ingest/text",
                    data={
                        "text": "This is a test document with enough content to be useful. " * 10,
                        "filename": "test.txt",
                    },
                )
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 201
        data = response.json()
        assert data["chunks_created"] == 3
        assert data["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_stats_endpoint(self):
        """GET /pipeline/stats should return collection info."""
        from httpx import AsyncClient, ASGITransport
        from backend.main import app

        self.mock_pipeline.vector_store.get_collection_stats.return_value = {
            "collection_name": "verirag_docs",
            "document_count": 42,
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/pipeline/stats")

        assert response.status_code == 200
        data = response.json()
        assert "document_count" in data
        assert "collection_name" in data


# ── Health Check Tests ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check_returns_200(self):
        """Health endpoint should return 200 with status field (even when degraded)."""
        from httpx import AsyncClient, ASGITransport
        from backend.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")
