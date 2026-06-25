"""routers/health.py - Health check and system readiness endpoint."""

from fastapi import APIRouter
from sqlalchemy import text

from backend.config import get_settings
from backend.schemas import HealthResponse

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check() -> HealthResponse:
    """
    Checks connectivity to PostgreSQL and ChromaDB.
    Returns 200 with status='ok' or status='degraded'.
    Only returns 503 if the process itself is broken (caught by the framework).

    This design lets load balancers distinguish 'app running but misconfigured'
    from 'app crashed' - both valid signals, but different responses.
    """
    db_status = "not configured"
    chroma_status = "not configured"

    # Check PostgreSQL
    if settings.database_url:
        try:
            from backend.database import get_db_context
            async with get_db_context() as db:
                await db.execute(text("SELECT 1"))
            db_status = "ok"
        except Exception as e:
            db_status = f"error: {str(e)[:120]}"

    # Check ChromaDB
    try:
        from backend.rag.vectorstore import get_vector_store
        vs = get_vector_store()
        stats = vs.get_collection_stats()
        chroma_status = f"ok (docs={stats['document_count']})"
    except Exception as e:
        chroma_status = f"error: {str(e)[:120]}"

    overall = "ok" if (db_status == "ok" and chroma_status.startswith("ok")) else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        database=db_status,
        chromadb=chroma_status,
        environment=settings.app_env,
    )
