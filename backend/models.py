"""models.py â€” SQLAlchemy ORM models for VeriRAG (Phase 2)."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    ARRAY, TIMESTAMP, Boolean, VARCHAR,
    Float, ForeignKey, Integer, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_tag: Mapped[str] = mapped_column(VARCHAR(100), nullable=False, index=True)
    pipeline_name: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    status: Mapped[str] = mapped_column(VARCHAR(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)

    avg_faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)

    run_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Phase 2: regression tracking
    has_regression: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    regression_details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    compared_to_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("eval_runs.id", ondelete="SET NULL", use_alter=True, name="fk_eval_run_compared_to"),
        nullable=True,
    )
    langsmith_run_url: Mapped[str | None] = mapped_column(VARCHAR(500), nullable=True)

    cases: Mapped[list["EvalCase"]] = relationship(
        "EvalCase", back_populates="eval_run",
        cascade="all, delete-orphan", lazy="selectin",
    )


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    contexts: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)

    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=utcnow)
    eval_run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="cases")


class PipelineDocument(Base):
    __tablename__ = "pipeline_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[str] = mapped_column(VARCHAR(500), nullable=False)
    filename: Mapped[str] = mapped_column(VARCHAR(500), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=utcnow)
    collection_name: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (UniqueConstraint("doc_id", "collection_name", name="uq_doc_collection"),)