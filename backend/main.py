"""
main.py - FastAPI application entry point.

Startup sequence:
1. Configure structured logging (loguru)
2. Warm up singletons (VectorStoreManager, RAGPipeline, RagasRunner)
3. Verify DB connectivity (non-fatal — app starts even if DB is unavailable)
4. Log configuration summary

The app starts successfully even when environment variables are missing.
Features degrade gracefully: endpoints that require unconfigured services
return HTTP 503 with a clear message rather than crashing the process.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from contextlib import asynccontextmanager

# Vercel's @vercel/python ASGI handler installs uvloop when it's importable.
# uvloop's uv_tcp_connect returns UV_EBUSY on SSL connections to Supabase
# in the Lambda environment. Force the standard asyncio policy before the
# event loop is created so we always use SelectorEventLoop on Vercel.
if os.environ.get("VERCEL"):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# ── Boot-error capture ────────────────────────────────────────────────────────
# Import the settings/db layer here so any ValidationError is visible before
# the first request arrives. Failures are stored and surfaced via /diag.
_BOOT_ERROR: str | None = None
_settings = None

try:
    from loguru import logger
    from sqlalchemy import text
    from backend.config import get_settings
    from backend.database import get_db_context, get_engine
    _settings = get_settings()
except Exception:
    _BOOT_ERROR = traceback.format_exc()

# Vercel sets VERCEL=1 in the serverless runtime
_IS_VERCEL = bool(os.environ.get("VERCEL"))


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    if _settings is None:
        return
    try:
        logger.remove()
        if _settings.is_production:
            logger.add(
                sys.stdout,
                format="{time:ISO8601} | {level} | {name}:{line} | {message}",
                level=_settings.log_level,
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
                level=_settings.log_level,
                colorize=True,
            )
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
    except Exception:
        pass


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()

    if _BOOT_ERROR:
        try:
            logger.error(f"Boot error captured:\n{_BOOT_ERROR}")
        except Exception:
            print(f"BOOT ERROR: {_BOOT_ERROR}", file=sys.stderr)
        yield
        return

    s = _settings
    try:
        logger.info("=" * 60)
        logger.info(f"VeriRAG starting | env={s.app_env} | vercel={_IS_VERCEL}")

        if not s.groq_api_key:
            logger.warning("GROQ_API_KEY not set — LLM/eval features unavailable")
        if not s.database_url:
            logger.warning("DATABASE_URL not set — database features unavailable")
        if not s.hf_token:
            logger.warning("HF_TOKEN not set — embeddings use anonymous rate limits")
    except Exception:
        pass

    # Warm up singletons — failures are non-fatal
    try:
        from backend.rag.vectorstore import get_vector_store
        get_vector_store()
        logger.info("VectorStore initialized")
    except Exception as e:
        logger.error(f"VectorStore init failed (non-fatal): {e}")

    try:
        from backend.rag.pipeline import get_pipeline
        get_pipeline()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.error(f"RAG pipeline init failed (non-fatal): {e}")

    try:
        from backend.evaluator.ragas_runner import get_ragas_runner
        get_ragas_runner()
        logger.info("RAGAS runner initialized")
    except Exception as e:
        logger.error(f"RAGAS runner init failed (non-fatal): {e}")

    # Create tables then verify connectivity (both non-fatal)
    if s.database_url:
        try:
            from backend.database import Base
            engine = get_engine()
            # create_all is idempotent — safe to call on every cold start.
            # Runs in the SAME event loop (no nested asyncio.run / EBUSY race).
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database schema ready (create_all applied)")
        except Exception as e:
            logger.error(f"Schema creation failed (non-fatal): {e}")

        try:
            async with get_db_context() as db:
                await db.execute(text("SELECT 1"))
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.error(f"Database connection failed (non-fatal): {e}")
    else:
        logger.info("Skipping DB check — DATABASE_URL not configured")

    try:
        logger.info("VeriRAG startup complete")
        logger.info("=" * 60)
    except Exception:
        pass

    yield

    # Shutdown
    try:
        logger.info("VeriRAG shutting down...")
        e = get_engine()
        await e.dispose()
        logger.info("Database pool disposed")
    except Exception:
        pass


# ── Application (must be module-level for @vercel/python static analysis) ─────

app = FastAPI(
    title="VeriRAG",
    description="""
## VeriRAG — Production RAG Evaluation & Observability Platform

VeriRAG evaluates RAG pipeline quality using [RAGAS](https://docs.ragas.io) metrics:

| Metric | Measures |
|--------|----------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Are useful chunks ranked higher in retrieval? |
| **Context Recall** | Does retrieved context cover all ground truth information? |

### Quick Start
1. `POST /api/v1/pipeline/ingest` — upload your documents
2. `POST /api/v1/eval/run/sample` — run a built-in sample evaluation
3. `GET /api/v1/eval/runs/{id}` — retrieve RAGAS scores

### Required Environment Variables
- `GROQ_API_KEY` — Groq API key (free at console.groq.com)
- `DATABASE_URL` — PostgreSQL connection string (postgresql+asyncpg://...)
- `HF_TOKEN` — HuggingFace token for embedding API
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ─────────────────────────────────────────────────────────────────

_cors_origins = ["*"]
if _settings is not None:
    try:
        _cors_origins = _settings.allowed_origins
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers (lazy import to avoid top-level import failures) ──────────────────

try:
    from backend.routers import health, eval, pipeline  # noqa: E402
    _api_prefix = _settings.api_v1_prefix if _settings else "/api/v1"
    app.include_router(health.router)
    app.include_router(eval.router, prefix=_api_prefix)
    app.include_router(pipeline.router, prefix=_api_prefix)
except Exception as _router_err:
    _boot_router_error = traceback.format_exc()
    if _BOOT_ERROR is None:
        globals()['_BOOT_ERROR'] = _boot_router_error


# ── Diagnostic endpoint (always available — exposes boot errors) ──────────────

@app.get("/diag", include_in_schema=False)
async def diag():
    """Diagnostic endpoint: exposes boot/import errors and live DB probe."""
    import re
    from urllib.parse import urlparse
    env_snapshot = {
        k: ("set" if v else "empty")
        for k, v in os.environ.items()
        if k in (
            "DATABASE_URL", "GROQ_API_KEY", "HF_TOKEN",
            "LANGCHAIN_TRACING_V2", "CHROMA_PERSIST_DIR",
            "APP_ENV", "VERCEL", "VERCEL_ENV", "LANGCHAIN_API_KEY",
        )
    }
    db_url_raw = os.environ.get("DATABASE_URL", "")
    # Parse host/port safely (handles passwords containing @)
    try:
        parsed = urlparse(db_url_raw)
        db_url_host = f"{parsed.scheme}://***:***@{parsed.hostname}:{parsed.port}{parsed.path}"
    except Exception:
        db_url_host = "parse error"

    # Live DB probe — full traceback on failure
    db_probe = "skipped"
    if db_url_raw and _settings:
        try:
            from backend.database import get_engine
            from sqlalchemy import text
            engine = get_engine()
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT version()"))
                db_probe = result.scalar()
        except Exception as probe_exc:
            db_probe = traceback.format_exc()

    # Report active event loop type to detect if uvloop is being used
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    loop_type = type(loop).__module__ + "." + type(loop).__name__

    # Report whether uvloop is importable (transitive dep check)
    try:
        import uvloop as _uvloop
        uvloop_info = f"installed v{getattr(_uvloop, '__version__', 'unknown')}"
    except ImportError:
        uvloop_info = "not installed"

    return {
        "boot_ok": _BOOT_ERROR is None,
        "boot_error": _BOOT_ERROR,
        "env": env_snapshot,
        "db_url_host": db_url_host,
        "db_probe": db_probe,
        "python_version": sys.version,
        "event_loop": loop_type,
        "uvloop": uvloop_info,
    }


# ── System Status ─────────────────────────────────────────────────────────────

@app.get("/api/v1/system/status", include_in_schema=False)
async def system_status():
    """Configuration status — used by frontend to show setup guide."""
    if _BOOT_ERROR:
        return {
            "configured": False,
            "missing_vars": ["BOOT_ERROR"],
            "environment": os.environ.get("APP_ENV", "unknown"),
            "version": "1.0.0",
            "boot_error": _BOOT_ERROR[:500],
        }

    s = _settings
    missing = []
    if not s.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not s.database_url:
        missing.append("DATABASE_URL")
    # HF_TOKEN is optional — embeddings work anonymously with stricter rate limits

    warnings = []
    if not s.hf_token:
        warnings.append("HF_TOKEN")

    return {
        "configured": len(missing) == 0,
        "missing_vars": missing,
        "optional_missing": warnings,
        "environment": s.app_env,
        "version": "1.0.0",
        "setup_guide": "https://github.com/OmNarkar777/Verirag#deployment" if missing else None,
    }


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    prefix = _settings.api_v1_prefix if _settings else "/api/v1"
    return {
        "service": "VeriRAG",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": prefix,
        "status": "/api/v1/system/status",
        "diag": "/diag",
    }
