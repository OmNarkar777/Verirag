"""
database.py - Async SQLAlchemy engine, session factory, and declarative base.

Engine is created lazily on first use so the module can be imported without
DATABASE_URL being set (enabling graceful degradation when env vars are missing).

Driver selection
────────────────
Local dev:    asyncpg  (fast, zero-config)
Vercel Lambda: psycopg3 SYNC in a per-session ThreadPoolExecutor

Why not asyncpg on Vercel:
  uvloop 0.22.1 pre-installed by Vercel Lambda uses libuv-native TCP handles
  for ALL asyncio loop.create_connection() calls (both SSL and plain TCP).
  When a handle is freed (uv_close) the callback is deferred to the next loop
  iteration.  Before that callback fires, uvloop may re-issue the same libuv
  handle memory to the next create_connection() call, which sees a handle
  already in CONNECTING state → UV_EBUSY ([Errno 16]).

  NOTE: asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy()) in
  main.py prevents uvloop from being installed, so the raw asyncpg path might
  also work — but we keep psycopg3 SYNC for safety.

Why psycopg3 SYNC works:
  Sync psycopg3 uses the C libpq library directly, NOT asyncio or libuv.
  libpq creates regular blocking sockets via the OS syscall layer — completely
  isolated from the event loop.  No EBUSY, no SSL path, no libuv.

URL parsing:
  DATABASE_URL passwords may contain '@' (common in Supabase auto-generated
  passwords).  SQLAlchemy's make_url() splits on the FIRST '@', corrupting the
  host component. We always parse via Python's urlparse (which uses the LAST '@')
  then build a SQLAlchemy URL.create() object to avoid this.

Threading model:
  Each _VercelSession wraps a sync psycopg3 Session in a dedicated
  ThreadPoolExecutor(max_workers=1).  Using one thread per session guarantees
  that the psycopg3 connection (which is not thread-safe) is always accessed
  from the same OS thread, even across multiple awaited execute() calls.
"""

import asyncio
import os
import re
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from backend.config import get_settings

settings = get_settings()

# ── shared globals ─────────────────────────────────────────────────────────────
_engine = None           # async asyncpg engine (local dev)
_session_factory = None  # async session factory (local dev)
_sync_engine = None      # sync psycopg3 engine (Vercel)
_sync_session_factory = None  # sync session factory (Vercel)

_CLOUD_HOSTS = ("supabase.com", "neon.tech", "neon.database.azure.com", "render.com")
_POOLER_MARKERS = ("pooler.supabase.com", ":6543")


# ── URL parsing ───────────────────────────────────────────────────────────────

def _parse_url_parts(raw: str) -> dict:
    """
    Parse DATABASE_URL using Python's urlparse which uses the LAST '@' as the
    userinfo/host boundary — correctly handling passwords that contain '@'.

    SQLAlchemy's make_url() splits on the FIRST '@', corrupting the host when
    the password contains '@' (common in Supabase auto-generated passwords).
    """
    from urllib.parse import urlparse, unquote
    p = urlparse(raw)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5432,
        "database": (p.path or "/postgres").lstrip("/") or "postgres",
        "username": unquote(p.username or ""),
        "password": unquote(p.password or ""),
    }


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
    url = re.sub(r"[?&]sslmode=\w+", "", url)
    for prefix in (
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
        "postgresql+psycopg://",
        "postgresql://",
        "postgres://",
    ):
        if url.startswith(prefix):
            return f"postgresql+{driver}://" + url[len(prefix):]
    return url


def _clean_url(url: str) -> str:
    return _to_driver_url(url, "asyncpg")


def _ensure_asyncpg_scheme(url: str) -> str:
    return _clean_url(url)


# ── connect_args ──────────────────────────────────────────────────────────────

def _asyncpg_connect_args(url: str) -> dict:
    args: dict = {}
    if any(h in url for h in _CLOUD_HOSTS):
        args["ssl"] = "require"
        if any(m in url for m in _POOLER_MARKERS):
            args["statement_cache_size"] = 0
    return args


def _psycopg_connect_args(url: str) -> dict:
    """
    psycopg3 connect_args for cloud/pooler hosts.
    sslmode=require  → libpq-native SSL (C library, not asyncio, not libuv)
    prepare_threshold=None → disable prepared statements for pgBouncer
    """
    args: dict = {}
    if any(h in url for h in _CLOUD_HOSTS):
        args["sslmode"] = "require"
        if any(m in url for m in _POOLER_MARKERS):
            args["prepare_threshold"] = None
    return args


# ── Vercel: sync psycopg3 engine ──────────────────────────────────────────────

def _get_sync_engine():
    """Sync SQLAlchemy engine using psycopg3 (C libpq, no asyncio).  Vercel only."""
    global _sync_engine
    if _sync_engine is not None:
        return _sync_engine

    raw = settings.database_url
    if not raw:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Add it in your Vercel environment variables (Supabase Transaction Pooler, port 6543)."
        )

    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL as SAURL

    # Parse with urlparse (handles '@' in password) then build URL object to
    # bypass SQLAlchemy's make_url() which splits on the FIRST '@', corrupting
    # the host when the password contains '@'.
    parts = _parse_url_parts(raw)
    sa_url = SAURL.create(
        drivername="postgresql+psycopg",
        username=parts["username"],
        password=parts["password"],
        host=parts["host"],
        port=parts["port"],
        database=parts["database"],
    )

    is_cloud = any(h in (parts["host"] or "") for h in _CLOUD_HOSTS)
    is_pooler = parts["port"] == 6543 or "pooler" in (parts["host"] or "")
    connect_args: dict = {}
    if is_cloud:
        connect_args["sslmode"] = "require"  # Supabase requires SSL; libpq C-native, no libuv
    if is_pooler:
        connect_args["prepare_threshold"] = None  # pgBouncer: disable prepared statements
    connect_args["connect_timeout"] = 10

    _sync_engine = create_engine(
        sa_url,
        poolclass=NullPool,      # pgBouncer pools server-side; no client pool needed
        connect_args=connect_args,
        echo=False,
    )
    return _sync_engine


def _get_sync_session_factory():
    global _sync_session_factory
    if _sync_session_factory is not None:
        return _sync_session_factory

    from sqlalchemy.orm import sessionmaker
    _sync_session_factory = sessionmaker(
        bind=_get_sync_engine(),
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    return _sync_session_factory


def create_all_sync() -> None:
    """
    Create all ORM tables using sync psycopg3.
    Called from the lifespan on Vercel via asyncio.to_thread() so the
    blocking libpq call doesn't stall the event loop.
    """
    engine = _get_sync_engine()
    Base.metadata.create_all(engine)
    engine.dispose()


# ── _VercelSession — async interface over a sync psycopg3 session ─────────────

class _VercelSession:
    """
    Wraps a synchronous SQLAlchemy Session in an async-compatible interface.

    All I/O methods are dispatched to a dedicated single-worker ThreadPoolExecutor
    so the blocking psycopg3 calls don't stall FastAPI's event loop.
    max_workers=1 guarantees the psycopg3 connection is always used from the
    same OS thread (psycopg3 connections are not thread-safe).
    """

    def __init__(self, sync_session) -> None:
        self._s = sync_session
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verirag-db")

    async def _io(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    # I/O operations — dispatched to the DB thread
    async def execute(self, stmt, params=None):
        if params is not None:
            return await self._io(lambda: self._s.execute(stmt, params))
        return await self._io(lambda: self._s.execute(stmt))

    async def scalar(self, stmt):
        return await self._io(lambda: self._s.scalar(stmt))

    async def scalars(self, stmt):
        return await self._io(lambda: self._s.scalars(stmt))

    async def get(self, entity, pk, **kwargs):
        return await self._io(lambda: self._s.get(entity, pk, **kwargs))

    async def flush(self, objects=None):
        await self._io(lambda: self._s.flush(objects))

    async def commit(self):
        await self._io(self._s.commit)

    async def rollback(self):
        await self._io(self._s.rollback)

    async def refresh(self, obj, attribute_names=None):
        await self._io(lambda: self._s.refresh(obj, attribute_names))

    async def close(self):
        try:
            await self._io(self._s.close)
        finally:
            self._executor.shutdown(wait=False)

    # Non-I/O operations — synchronous (mirrors AsyncSession interface)
    def add(self, instance, _warn: bool = True) -> None:
        self._s.add(instance, _warn=_warn)

    def add_all(self, instances) -> None:
        self._s.add_all(instances)

    async def delete(self, instance) -> None:
        # Marking for deletion is sync in SQLAlchemy; expose as async to
        # match AsyncSession.delete() which callers already await.
        self._s.delete(instance)

    async def merge(self, instance):
        return await self._io(lambda: self._s.merge(instance))


# ── Async asyncpg engine (local dev) ──────────────────────────────────────────

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

    # Use URL.create() so passwords containing '@' are handled correctly
    from sqlalchemy.engine import URL as SAURL
    parts = _parse_url_parts(raw)
    sa_url = SAURL.create(
        drivername="postgresql+asyncpg",
        username=parts["username"],
        password=parts["password"],
        host=parts["host"],
        port=parts["port"],
        database=parts["database"],
    )

    is_cloud = any(h in (parts["host"] or "") for h in _CLOUD_HOSTS)
    is_pooler = parts["port"] == 6543 or "pooler" in (parts["host"] or "")
    connect_args: dict = {}
    if is_cloud:
        connect_args["ssl"] = "require"
    if is_pooler:
        connect_args["statement_cache_size"] = 0

    _engine = create_async_engine(
        sa_url,
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


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator:
    """
    FastAPI dependency that yields a database session.

    Vercel: yields a _VercelSession (sync psycopg3 in ThreadPoolExecutor).
    Local:  yields an AsyncSession (async asyncpg).

    Both raise HTTP 503 on DB errors so the frontend shows a clear message.
    """
    from fastapi import HTTPException
    from sqlalchemy.exc import OperationalError, SQLAlchemyError

    if os.environ.get("VERCEL"):
        try:
            factory = _get_sync_session_factory()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        sync_session = factory()
        session = _VercelSession(sync_session)
        try:
            yield session
            await session.commit()
        except HTTPException:
            raise
        except (OperationalError, SQLAlchemyError) as e:
            await session.rollback()
            raise HTTPException(
                status_code=503,
                detail=f"Database unavailable. Check DATABASE_URL. ({type(e).__name__})",
            )
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
        return

    # ── Local async path ──────────────────────────────────────────────────────
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
                        f"Check DATABASE_URL in environment variables. ({type(e).__name__})"
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
            detail=f"Cannot connect to database. ({type(e).__name__})",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach database host. ({type(e).__name__})",
        )


@asynccontextmanager
async def get_db_context() -> AsyncGenerator:
    """Context manager version of get_db for use outside of FastAPI DI."""
    if os.environ.get("VERCEL"):
        factory = _get_sync_session_factory()
        sync_session = factory()
        session = _VercelSession(sync_session)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
        return

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
