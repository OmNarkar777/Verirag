"""
database.py - Async SQLAlchemy engine, session factory, and declarative base.

Engine is created lazily on first use so the module can be imported without
DATABASE_URL being set (enabling graceful degradation when env vars are missing).

Driver selection:
- Local dev: asyncpg (fast, zero config)
- Vercel Lambda: psycopg3 (psycopg[asyncio])
    uvloop 0.22 pre-installed by Vercel uses libuv-native SSL in
    create_connection(ssl=ctx), which returns UV_EBUSY on Lambda.
    psycopg3 uses asyncio.start_tls() which goes through Python's ssl
    module and does not hit the libuv SSL path — no EBUSY.
"""

import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings

settings = get_settings()

_engine = None
_session_factory = None

_CLOUD_HOSTS = ("supabase.com", "neon.tech", "neon.database.azure.com", "render.com")
_POOLER_MARKERS = ("pooler.supabase.com", ":6543")


# ── URL normalisation ─────────────────────────────────────────────────────────

def _to_driver_url(url: str, driver: str) -> str:
    """
    Normalise any postgresql:// variant to use the requested driver.

    Handles:
        postgresql://          (Supabase default — no driver)
        postgres://            (shorthand)
        postgresql+asyncpg://  (explicit asyncpg)
        postgresql+psycopg2:// (legacy)
        postgresql+psycopg://  (psycopg3)
    """
    # Strip sslmode= — both drivers configure SSL via connect_args, not URL
    url = re.sub(r"[?&]sslmode=\w+", "", url)
    # Normalise scheme
    for prefix in (
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
        "postgresql+psycopg://",
        "postgresql://",
        "postgres://",
    ):
        if url.startswith(prefix):
            return f"postgresql+{driver}://" + url[len(prefix):]
    return url  # already correct or unknown scheme


def _clean_url(url: str) -> str:
    """Normalise to asyncpg scheme (local default).  Strips sslmode."""
    return _to_driver_url(url, "asyncpg")


def _ensure_asyncpg_scheme(url: str) -> str:
    """Alias kept for any external callers."""
    return _clean_url(url)


# ── connect_args per driver ───────────────────────────────────────────────────

def _asyncpg_connect_args(url: str) -> dict:
    """
    asyncpg connect_args for cloud hosts.
    ssl="require" → asyncpg creates an isolated SSLContext per connection.
    statement_cache_size=0 → required for pgBouncer Transaction Pooler
    (asyncpg prepared stmts are per server-connection; pgBouncer multiplexes).
    """
    args: dict = {}
    if any(h in url for h in _CLOUD_HOSTS):
        args["ssl"] = "require"
        if any(m in url for m in _POOLER_MARKERS):
            args["statement_cache_size"] = 0
    return args


def _psycopg_connect_args(url: str) -> dict:
    """
    psycopg3 connect_args for cloud hosts.
    sslmode="require" → libpq-style SSL (psycopg3 passes to start_tls).
    prepare_threshold=None → disable server-side prepared statements for
    pgBouncer Transaction Pooler (equivalent to asyncpg statement_cache_size=0).
    """
    args: dict = {}
    if any(h in url for h in _CLOUD_HOSTS):
        args["sslmode"] = "require"
        if any(m in url for m in _POOLER_MARKERS):
            args["prepare_threshold"] = None
    return args


# ── Engine factory ────────────────────────────────────────────────────────────

def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    raw = settings.database_url
    if not raw:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Add it in your Vercel project environment variables. "
            "Use the Supabase Transaction Pooler URL (port 6543), e.g.: "
            "postgresql://postgres.XXXX:PASS@aws-0-REGION.pooler.supabase.com:6543/postgres"
        )

    if os.environ.get("VERCEL"):
        # uvloop 0.22.1 on Vercel Lambda returns UV_EBUSY for ALL SSL
        # connections — both asyncpg (create_connection(ssl=ctx)) and
        # psycopg3 (start_tls) route through uvloop's libuv SSL which is
        # broken.  Plain TCP (ssl=False / sslmode=disable) bypasses the
        # SSL path entirely; uvloop's TCP-only connect works fine.
        # Supabase pgBouncer Transaction Pooler accepts non-SSL clients
        # (server sets client_tls_sslmode=allow by default).
        # NullPool: pgBouncer handles connection pooling server-side.
        url = _to_driver_url(raw, "asyncpg")
        connect_args: dict = {
            "ssl": False,
            "statement_cache_size": 0,  # required for pgBouncer tx pooler
        }
        from sqlalchemy.pool import NullPool
        _engine = create_async_engine(
            url,
            poolclass=NullPool,
            connect_args=connect_args,
            echo=False,
        )
    else:
        # Local / non-Vercel: asyncpg is faster and has no uvloop issue.
        url = _clean_url(raw)
        connect_args = _asyncpg_connect_args(url)
        _engine = create_async_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args=connect_args,
            echo=not settings.is_production,
        )

    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.
    Raises HTTP 503 (not 500) so the frontend shows a setup message
    instead of a generic error.
    """
    from fastapi import HTTPException
    from sqlalchemy.exc import OperationalError, SQLAlchemyError

    try:
        factory = _get_session_factory()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except HTTPException:
                raise
            except (OperationalError, SQLAlchemyError) as e:
                await session.rollback()
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Database unavailable. "
                        f"Check DATABASE_URL in Vercel environment variables. ({type(e).__name__})"
                    ),
                )
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except HTTPException:
        raise
    except (OperationalError, SQLAlchemyError) as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to database. "
                f"Check DATABASE_URL in Vercel environment variables. ({type(e).__name__})"
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot reach database host. "
                f"Check DATABASE_URL in Vercel environment variables. ({type(e).__name__})"
            ),
        )


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager version of get_db for use outside of FastAPI DI."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
