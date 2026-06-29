"""schemas.py â€” Pydantic v2 request/response schemas."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TestCaseInput(BaseModel):
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)

    @field_validator("contexts")
    @classmethod
    def contexts_must_have_content(cls, v: list[str]) -> list[str]:
        if any(not ctx.strip() for ctx in v):
            raise ValueError("All context strings must be non-empty")
        return v


class EvalRunRequest(BaseModel):
    version_tag: str = Field(..., pattern=r"^v\d+\.\d+\.\d+.*$")
    pipeline_name: str = Field(..., min_length=1, max_length=200)
    test_cases: list[TestCaseInput] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricScores(BaseModel):
    faithfulness: float | None = Field(None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(None, ge=0.0, le=1.0)
    context_precision: float | None = Field(None, ge=0.0, le=1.0)
    context_recall: float | None = Field(None, ge=0.0, le=1.0)


class EvalCaseResult(BaseModel):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness_score: float | None
    answer_relevancy_score: float | None
    context_precision_score: float | None
    context_recall_score: float | None
    created_at: datetime


class EvalRunResponse(BaseModel):
    eval_run_id: uuid.UUID
    version_tag: str
    status: str
    message: str = "Evaluation started in background"


class EvalRunSummary(BaseModel):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    version_tag: str
    pipeline_name: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    total_cases: int
    scores: MetricScores | None = None


class EvalRunDetail(EvalRunSummary):
    cases: list[EvalCaseResult] = Field(default_factory=list)
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


# â”€â”€ Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(default=5, ge=1, le=20)
    collection_name: str | None = None
    # Retrieval experiment parameters — allow UI-driven experimentation
    # without code changes. These override the pipeline defaults.
    use_mmr: bool = Field(default=True, description="MMR retrieval (True) vs cosine similarity (False)")
    fetch_k: int = Field(default=20, ge=5, le=100, description="Candidate pool size for MMR")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR relevance weight (1.0=pure relevance, 0.0=pure diversity)")


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    model_used: str
    langsmith_trace_url: str | None = None
    confidence: str = Field(default="medium", description="high | medium | low based on top chunk similarity")


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    collection_name: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    chromadb: str
    environment: str


class PaginatedCases(BaseModel):
    total: int
    page: int
    page_size: int
    cases: list[EvalCaseResult]


# â”€â”€ Regression (Phase 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class MetricDelta(BaseModel):
    previous: float
    current: float
    delta: float
    threshold: float
    is_regression: bool


class RegressionSummary(BaseModel):
    has_regression: bool
    compared_to_run_id: Any | None = None
    metrics: dict[str, MetricDelta] = Field(default_factory=dict)


class EvalRunWithRegression(EvalRunSummary):
    """EvalRunSummary extended with regression info."""
    has_regression: bool = False
    regression_details: dict[str, Any] = Field(default_factory=dict)
    compared_to_run_id: Any | None = None
    langsmith_run_url: str | None = None