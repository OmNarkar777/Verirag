"""
routers/pipeline.py — RAG pipeline ingest and query endpoints.

ENDPOINTS:
POST /pipeline/ingest  → upload text or PDF to ChromaDB
POST /pipeline/query   → query the RAG pipeline, get answer + chunks
GET  /pipeline/stats   → collection stats
"""
from __future__ import annotations

import tempfile
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import PipelineDocument
from backend.rag.pipeline import RAGPipeline, get_pipeline
from backend.schemas import IngestResponse, QueryRequest, QueryResponse, RetrievedChunk
from backend.services.langsmith_service import LangSmithService, get_langsmith_service

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=201,
    summary="Ingest a document into ChromaDB",
    description="""
    Upload a text or PDF file to be chunked and indexed in ChromaDB.
    The document will be chunked (512 tokens, 50 overlap), embedded with
    all-MiniLM-L6-v2, and stored for retrieval.

    Ingestion is tracked in PostgreSQL for audit and cross-reference with eval runs.
    Re-uploading the same file is idempotent (upsert behavior).
    """,
)
async def ingest_document(
    file: UploadFile = File(..., description="Text (.txt) or PDF (.pdf) file"),
    collection_name: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """
    POST /pipeline/ingest
    
    Handles both .txt and .pdf files.
    Tracks ingestion in PostgreSQL for audit trail.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    filename = file.filename.lower()
    allowed_types = {".txt", ".pdf", ".md"}
    ext = os.path.splitext(filename)[1]

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}",
        )

    content = await file.read()

    try:
        if ext == ".pdf":
            # Write to temp file for PyPDFLoader (requires file path, not bytes)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = pipeline.ingest_pdf(
                    file_path=tmp_path,
                    collection_name=collection_name,
                )
            finally:
                os.unlink(tmp_path)  # always clean up temp file
        else:
            # Text files: decode and ingest directly
            text = content.decode("utf-8", errors="replace")
            result = pipeline.ingest_text(
                text=text,
                filename=file.filename,
                collection_name=collection_name,
            )

        # Track in PostgreSQL — enables queries like "what's in the vector store?"
        doc_record = PipelineDocument(
            doc_id=result["doc_id"],
            filename=file.filename,
            chunk_count=result["chunks_created"],
            collection_name=result["collection_name"],
            doc_metadata={
                "file_type": ext,
                "file_size_bytes": len(content),
            },
        )
        db.add(doc_record)

        logger.info(
            f"Document ingested | filename={file.filename} | "
            f"chunks={result['chunks_created']}"
        )

        return IngestResponse(
            doc_id=result["doc_id"],
            filename=file.filename,
            chunks_created=result["chunks_created"],
            collection_name=result["collection_name"],
            message=f"Successfully ingested '{file.filename}' into {result['chunks_created']} chunks",
        )

    except Exception as e:
        logger.error(f"Ingestion failed | filename={file.filename} | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=201,
    summary="Ingest raw text directly (no file upload)",
)
async def ingest_text_direct(
    text: str = Form(..., min_length=10),
    filename: str = Form(..., description="Identifier for this text"),
    collection_name: str | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """
    POST /pipeline/ingest/text
    
    Convenience endpoint for ingesting text without file upload.
    Useful for programmatic ingestion of documents during tests.
    """
    try:
        result = pipeline.ingest_text(
            text=text,
            filename=filename,
            collection_name=collection_name,
        )

        doc_record = PipelineDocument(
            doc_id=result["doc_id"],
            filename=filename,
            chunk_count=result["chunks_created"],
            collection_name=result["collection_name"],
            doc_metadata={"source": "direct_text", "text_length": len(text)},
        )
        db.add(doc_record)

        return IngestResponse(
            doc_id=result["doc_id"],
            filename=filename,
            chunks_created=result["chunks_created"],
            collection_name=result["collection_name"],
            message=f"Ingested text as '{filename}' into {result['chunks_created']} chunks",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
    description="""
    Query the RAG pipeline with a natural language question.
    Returns the generated answer plus the retrieved chunks used to generate it.

    The retrieved chunks are EXACTLY what RAGAS evaluates — they become the
    `contexts` field in an eval test case.
    """,
)
async def query_pipeline(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    langsmith: LangSmithService = Depends(get_langsmith_service),
) -> QueryResponse:
    """
    POST /pipeline/query
    
    This endpoint generates the (question, answer, contexts) tuple that
    can be fed directly into POST /eval/run for evaluation.
    
    Tip: run queries → collect (question, answer, contexts) → add ground_truth
    → POST to /eval/run for a real-world eval set.
    """
    try:
        result = pipeline.query(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
        )

        chunks = [
            RetrievedChunk(
                content=chunk["content"],
                source=chunk["source"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {}),
            )
            for chunk in result["retrieved_chunks"]
        ]

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=chunks,
            model_used=result["model_used"],
            langsmith_trace_url=langsmith.get_project_url(),
            confidence=result.get("confidence", "medium"),
        )

    except Exception as e:
        logger.error(f"Pipeline query failed | question={request.question[:50]} | error={e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get(
    "/stats",
    summary="Get ChromaDB collection statistics",
)
async def get_pipeline_stats(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> dict:
    """GET /pipeline/stats — document count and collection info."""
    from backend.config import get_settings
    settings = get_settings()

    stats = pipeline.vector_store.get_collection_stats()
    return {
        "collection_name": stats["collection_name"],
        "document_count": stats["document_count"],
        "embedding_model": settings.embedding_model,
        "retrieval_strategy": "MMR",
        "top_k": settings.retrieval_top_k,
        "mmr_lambda": settings.retrieval_lambda,
    }


@router.get(
    "/documents",
    summary="List ingested documents from PostgreSQL",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
) -> list[dict]:
    """GET /pipeline/documents — audit trail of all ingested documents."""
    from sqlalchemy import select, desc
    from backend.models import PipelineDocument

    result = await db.execute(
        select(PipelineDocument)
        .order_by(desc(PipelineDocument.ingested_at))
        .limit(limit)
    )
    docs = result.scalars().all()

    return [
        {
            "id": str(doc.id),
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "chunk_count": doc.chunk_count,
            "collection_name": doc.collection_name,
            "ingested_at": doc.ingested_at.isoformat(),
            "metadata": doc.doc_metadata,
        }
        for doc in docs
    ]
