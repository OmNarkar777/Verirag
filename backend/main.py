"""
main.py - FastAPI application entry point.

Startup sequence:
1. Configure structured logging (loguru)
2. Warm up singletons (VectorStoreManager, RAGPipeline, RagasRunner)
3. Verify DB connectivity (non-fatal - app starts even if DB is unavailable)
4. Log configuration summary

The app starts successfully even when environment variables are missing.
Features degrade gracefully: endpoints that require unconfigured services
return HTTP 503 with a clear message rather than crashing the process.
"""

import os
import sys
import traceback
from contextlib import asynccontextmanager

_STARTUP_ERROR: str | None = None

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from loguru import logger
    from sqlalchemy import text

    from backend.config import get_settings
    from backend.database import get_db_context, get_engine

    settings = get_settings()
    _IMPORTS_OK = True
except Exception:
    _STARTUP_ERROR = traceback.format_exc()
    _IMPORTS_OK = False
    settings = None  # type: ignore[assignment]

# Vercel sets VERCEL=1 in its serverless runtime environment
_IS_VERCEL = bool(os.environ.get("VERCEL"))


def configure_logging() -> None:
    """Configure loguru for JSON output in production, colored output in dev."""
    logger.remove()

    if settings.is_production:
        logger.add(
            sys.stdout,
            format="{time:ISO8601} | {level} | {name}:{line} | {message}",
            level=settings.log_level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=settings.log_level,
            colorize=True,
        )

    # File logging: skip on Vercel (read-only FS); use /tmp on any other Linux env
    if not _IS_VERCEL:
        log_dir = "logs"
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.add(
                f"{log_dir}/verirag.log",
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                level="INFO",
                enqueue=True,
            )
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    configure_logging()

    logger.info("=" * 60)
    logger.info(f"VeriRAG starting | env={settings.app_env} | vercel={_IS_VERCEL}")

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set - LLM and evaluation features will be unavailable")
    if not settings.database_url:
        logger.warning("DATABASE_URL not set - database features will be unavailable")
    if not settings.hf_token:
        logger.warning("HF_TOKEN not set - embedding API calls will use anonymous rate limits")

    # Warm up singletons - failures are logged but do not prevent startup
    from backend.rag.vectorstore import get_vector_store
    from backend.rag.pipeline import get_pipeline
    from backend.evaluator.ragas_runner import get_ragas_runner

    try:
        get_vector_store()
        logger.info("VectorStore initialized")
    except Exception as e:
        logger.error(f"VectorStore init failed: {e}")

    try:
        get_pipeline()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.error(f"RAG pipeline init failed: {e}")

    try:
        get_ragas_runner()
        logger.info("RAGAS runner initialized")
    except Exception as e:
        logger.error(f"RAGAS runner init failed: {e}")

    # Verify DB connectivity (non-fatal)
    if settings.database_url:
        try:
            async with get_db_context() as db:
                await db.execute(text("SELECT 1"))
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
    else:
        logger.info("Skipping DB check - DATABASE_URL not configured")

    logger.info("VeriRAG startup complete")
    logger.info("=" * 60)

    yield  # ── Application runs ──────────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("VeriRAG shutting down...")
    try:
        e = get_engine()
        await e.dispose()
        logger.info("Database connection pool disposed")
    except Exception:
        pass
    logger.info("Shutdown complete.")


# ── FastAPI Application ────────────────────────────────────────────────────────

if not _IMPORTS_OK:
    # Surface the real boot error as a JSON response so Vercel logs capture it
    from fastapi import FastAPI as _FastAPI

    app = _FastAPI(title="VeriRAG [BOOT ERROR]")

    @app.get("/{path:path}", include_in_schema=False)
    async def _boot_error(path: str):
        return {"boot_error": _STARTUP_ERROR, "env_vars": {
            k: ("set" if v else "empty")
            for k, v in os.environ.items()
            if k in ("DATABASE_URL", "GROQ_API_KEY", "HF_TOKEN", "LANGCHAIN_TRACING_V2",
                     "CHROMA_PERSIST_DIR", "APP_ENV", "VERCEL", "VERCEL_ENV")
        }}

else:
    app = FastAPI(
        title="VeriRAG",
        description="""
## VeriRAG - Production RAG Evaluation & Observability Platform

VeriRAG evaluates RAG pipeline quality using [RAGAS](https://docs.ragas.io) metrics:

| Metric | Measures |
|--------|----------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Are useful chunks ranked higher in retrieval? |
| **Context Recall** | Does retrieved context cover all ground truth information? |

### Quick Start
1. `POST /api/v1/pipeline/ingest` - upload your documents
2. `POST /api/v1/eval/run/sample` - run built-in sample evaluation
3. `GET /api/v1/eval/runs/{id}` - retrieve RAGAS scores

### Setup
Set these environment variables in your deployment:
- `GROQ_API_KEY` - Groq API key (get free at console.groq.com)
- `DATABASE_URL` - PostgreSQL connection string (postgresql+asyncpg://...)
- `HF_TOKEN` - HuggingFace token for embedding API
    """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Middleware ─────────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routers ───────────────────────────────────────────────────────────────────

    from backend.routers import health, eval, pipeline  # noqa: E402

    app.include_router(health.router)
    app.include_router(eval.router, prefix=settings.api_v1_prefix)
    app.include_router(pipeline.router, prefix=settings.api_v1_prefix)

    # ── System Status ──────────────────────────────────────────────────────────────

    @app.get("/api/v1/system/status", include_in_schema=False)
    async def system_status():
        """Returns configuration status - used by frontend to show setup guide."""
        missing = []
        if not settings.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not settings.database_url:
            missing.append("DATABASE_URL")
        if not settings.hf_token:
            missing.append("HF_TOKEN")

        return {
            "configured": len(missing) == 0,
            "missing_vars": missing,
            "environment": settings.app_env,
            "version": "1.0.0",
            "setup_guide": "https://github.com/OmNarkar777/Verirag#deployment" if missing else None,
        }

    # ── Root ──────────────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "VeriRAG",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "api": settings.api_v1_prefix,
            "status": "/api/v1/system/status",
        }
