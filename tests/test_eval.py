"""
tests/test_eval.py — Async tests for evaluation endpoints and service.

TESTING STRATEGY:
- Use pytest-asyncio for async test functions
- Use httpx AsyncClient for endpoint tests (not TestClient which is sync)
- Mock RAGAS and Groq calls — we test OUR code, not RAGAS internals
- Use in-memory SQLite for tests — no PostgreSQL needed in CI
  (SQLAlchemy's async engine supports SQLite with aiosqlite)

WHAT WE TEST:
1. EvalRunRequest validation (Pydantic schemas)
2. POST /eval/run → returns 202 with correct structure
3. GET /eval/runs → returns list
4. GET /eval/runs/{id} → returns correct detail
5. Score persistence — stored scores match input
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.schemas import TestCaseInput as EvalTestCase


# ── Sample Test Data ───────────────────────────────────────────────────────────

SAMPLE_TEST_CASES = [
    EvalTestCase(
        question="What is retrieval-augmented generation?",
        answer="RAG combines retrieval of relevant documents with language model generation.",
        contexts=["RAG is a technique that retrieves documents before LLM generation."],
        ground_truth="RAG retrieves relevant documents from a knowledge base and uses them as context for LLM generation.",
    ),
    EvalTestCase(
        question="What does RAGAS measure?",
        answer="RAGAS measures faithfulness, answer relevancy, context precision, and context recall.",
        contexts=["RAGAS is a framework for evaluating RAG pipelines using four metrics."],
        ground_truth="RAGAS evaluates RAG systems using faithfulness, answer relevancy, context precision, and context recall metrics.",
    ),
]

SAMPLE_EVAL_REQUEST = {
    "version_tag": "v1.0.0-test",
    "pipeline_name": "test-pipeline",
    "test_cases": [tc.model_dump() for tc in SAMPLE_TEST_CASES],
    "metadata": {"chunk_size": 512, "top_k": 5},
}


# ── Schema Validation Tests ────────────────────────────────────────────────────

class TestEvalRunRequest:
    def test_valid_version_tag_formats(self):
        """Version tags must follow v{major}.{minor}.{patch} format."""
        from backend.schemas import EvalRunRequest

        valid_tags = ["v1.0.0", "v1.0.0-baseline", "v2.1.3-hybrid-mmr"]
        for tag in valid_tags:
            req = EvalRunRequest(
                version_tag=tag,
                pipeline_name="test",
                test_cases=SAMPLE_TEST_CASES,
            )
            assert req.version_tag == tag

    def test_invalid_version_tag_rejected(self):
        """Non-semver tags should raise validation error."""
        from backend.schemas import EvalRunRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            EvalRunRequest(
                version_tag="baseline",  # missing v prefix and semver
                pipeline_name="test",
                test_cases=SAMPLE_TEST_CASES,
            )
        assert "version_tag" in str(exc_info.value)

    def test_empty_test_cases_rejected(self):
        """Empty test cases list should fail validation."""
        from backend.schemas import EvalRunRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalRunRequest(
                version_tag="v1.0.0",
                pipeline_name="test",
                test_cases=[],
            )

    def test_empty_context_strings_rejected(self):
        """Contexts with empty strings should fail validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalTestCase(
                question="What is RAG?",
                answer="RAG is...",
                contexts=["valid context", ""],  # empty string should fail
                ground_truth="RAG retrieves documents.",
            )


# ── EvalService Unit Tests ────────────────────────────────────────────────────

class TestEvalService:
    @pytest.mark.asyncio
    async def test_list_eval_runs_empty(self):
        """Empty database should return empty list (uses mock session — avoids JSONB/SQLite incompatibility)."""
        from backend.services.eval_service import EvalService

        session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        service = EvalService()
        runs = await service.list_eval_runs(db=session)
        assert runs == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self):
        """Querying a non-existent run should return None."""
        from backend.services.eval_service import EvalService

        session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        service = EvalService()
        result = await service.get_eval_run(db=session, run_id=uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_score_classification(self):
        """Score classification thresholds should work correctly."""
        from backend.evaluator.metrics import classify_score

        assert classify_score("faithfulness", 0.90) == "pass"
        assert classify_score("faithfulness", 0.72) == "warning"
        assert classify_score("faithfulness", 0.50) == "fail"
        assert classify_score("unknown_metric", 0.90) == "unknown"

    @pytest.mark.asyncio
    async def test_score_summary_structure(self):
        """score_summary should return enriched dict with status and description."""
        from backend.evaluator.metrics import score_summary

        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.72,
            "context_precision": 0.60,
            "context_recall": 0.45,
        }
        summary = score_summary(scores)

        assert "faithfulness" in summary
        assert summary["faithfulness"]["score"] == 0.85
        assert summary["faithfulness"]["status"] == "pass"
        assert isinstance(summary["faithfulness"]["description"], str)

        assert summary["context_recall"]["status"] == "fail"


# ── Dataset Builder Tests ─────────────────────────────────────────────────────

class TestDatasetBuilder:
    def test_sample_test_cases_count(self):
        """Should return exactly 10 sample test cases."""
        from backend.evaluator.dataset_builder import get_sample_test_cases

        cases = get_sample_test_cases()
        assert len(cases) == 10

    def test_all_cases_have_required_fields(self):
        """Every case must have question, answer, contexts, ground_truth."""
        from backend.evaluator.dataset_builder import get_sample_test_cases

        cases = get_sample_test_cases()
        for case in cases:
            assert case.question
            assert case.answer
            assert len(case.contexts) > 0
            assert all(ctx.strip() for ctx in case.contexts)
            assert case.ground_truth

    def test_build_ragas_dataset_format(self):
        """RAGAS dataset should have correct column names and row count."""
        from backend.evaluator.dataset_builder import build_ragas_dataset

        cases = SAMPLE_TEST_CASES
        dataset = build_ragas_dataset(cases)

        assert set(dataset.column_names) == {"question", "answer", "contexts", "ground_truth"}
        assert len(dataset) == len(cases)

    def test_build_ragas_dataset_values(self):
        """Dataset values should match input test cases."""
        from backend.evaluator.dataset_builder import build_ragas_dataset

        cases = SAMPLE_TEST_CASES
        dataset = build_ragas_dataset(cases)

        assert dataset["question"][0] == cases[0].question
        assert dataset["answer"][0] == cases[0].answer
        assert dataset["contexts"][0] == cases[0].contexts
        assert dataset["ground_truth"][0] == cases[0].ground_truth


# ── API Endpoint Tests (with mocked RAGAS) ───────────────────────────────────

@pytest.fixture
def mock_eval_service():
    """
    Mock the eval service via dependency_overrides — correct way to mock
    FastAPI deps since Depends() captures the function reference at definition time,
    not at call time, so module-level patch() has no effect.
    """
    from backend.main import app
    from backend.services.eval_service import get_eval_service

    service = MagicMock()
    service.start_eval_run = AsyncMock(return_value=uuid.uuid4())
    service.execute_evaluation = AsyncMock(return_value=None)
    service.list_eval_runs = AsyncMock(return_value=[])
    service.get_eval_run = AsyncMock(return_value=None)

    app.dependency_overrides[get_eval_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_eval_service, None)


@pytest.fixture
def mock_db():
    """Override get_db so endpoint tests don't need a real database."""
    from backend.main import app
    from backend.database import get_db

    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        )
    )
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    async def _mock_get_db():
        yield session

    app.dependency_overrides[get_db] = _mock_get_db
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
class TestEvalEndpoints:
    async def test_post_eval_run_returns_202(self, mock_eval_service, mock_db):
        """POST /eval/run should return 202 Accepted with run_id."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/eval/run",
                json=SAMPLE_EVAL_REQUEST,
            )

        assert response.status_code == 202
        data = response.json()
        assert "eval_run_id" in data
        assert data["status"] == "running"
        assert data["version_tag"] == "v1.0.0-test"

    async def test_post_eval_run_invalid_version_returns_422(self, mock_eval_service, mock_db):
        """Invalid version tag should return 422 Unprocessable Entity."""
        bad_request = {**SAMPLE_EVAL_REQUEST, "version_tag": "not-semver"}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/eval/run", json=bad_request)

        assert response.status_code == 422

    async def test_get_eval_runs_returns_list(self, mock_db):
        """GET /eval/runs should return a list (no eval service needed for list endpoint)."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/eval/runs")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_nonexistent_run_returns_404(self, mock_eval_service, mock_db):
        """GET /eval/runs/{unknown_id} should return 404."""
        run_id = uuid.uuid4()
        mock_eval_service.get_eval_run = AsyncMock(return_value=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/v1/eval/runs/{run_id}")

        assert response.status_code == 404
