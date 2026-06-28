"""
database.py - Async SQLAlchemy engine, session factory, and declarative base.

Engine is created lazily on first use so the module can be imported without
DATABASE_URL being set (enabling graceful degradation when env vars are missing).
"""

import os
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


def _build_connect_args(url: str) -> dict:
    """
    Return asyncpg connect_args appropriate for the target database.

    Cloud PostgreSQL providers (Supabase, Neon, Render) require SSL.
    asyncpg ignores psycopg2-style `sslmode=require` URL params — the
    only valid approach is connect_args={"ssl": "require"}.

    pgBouncer Transaction Pooler (Supabase port 6543): asyncpg caches
    prepared statements per server connection.  pgBouncer multiplexes
    client connections across different server connections, so a prepared
    statement prepared on connection A may not exist on connection B.
    statement_cache_size=0 disables the cache and is required when
    connecting through any pgBouncer Transaction/Statement pooler.
    """
    cloud_hosts = ("supabase.com", "neon.tech", "neon.database.azure.com", "render.com")
    args: dict = {}
    if any(h in url for h in cloud_hosts):
        args["ssl"] = "require"
        # pgBouncer (Supabase Transaction/Session Pooler): disable prepared
        # statement cache so asyncpg doesn't assume a statement prepared on
        # one server connection exists on a different server connection.
        if "pooler.supabase.com" in url or ":6543" in url:
            args["statement_cache_size"] = 0
    return args


def _ensure_asyncpg_scheme(url: str) -> str:
    """
    Normalise any postgresql:// variant to postgresql+asyncpg://.

    Supabase and most cloud providers emit plain postgresql:// URIs.
    SQLAlchemy's create_async_engine falls back to psycopg2 (the sync
    default) when no driver is specified, raising ModuleNotFoundError
    if psycopg2 is not installed.  This normaliser converts:
      postgresql://      → postgresql+asyncpg://
      postgresql+psycopg2:// → postgresql+asyncpg://
    so the user can paste the URL from Supabase directly without editing it.
    """
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url.split("://", 1)[1]
    elif url.startswith("postgresql+psycopg2://"):
        url = "postgresql+asyncpg://" + url.split("://", 1)[1]
    return url


def _clean_url(url: str) -> str:
    """Remove psycopg2-style sslmode param — asyncpg ignores it and may warn."""
    import re
    url = _ensure_asyncpg_scheme(url)
    return re.sub(r"[?&]sslmode=\w+", "", url)


def get_engine():
    global _engine
    if _engine is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. "
                "Add it in your Vercel project environment variables. "
                "Example: postgresql+asyncpg://user:pass@host:5432/dbname"
            )
        url = _clean_url(settings.database_url)
        connect_args = _build_connect_args(url)
        if os.environ.get("VERCEL"):
            # Supabase Transaction Pooler (port 6543) is pgBouncer — it handles
            # connection pooling server-side.  Use NullPool on our side so each
            # serverless invocation gets a fresh connection through pgBouncer
            # rather than maintaining a persistent pool that conflicts with
            # pgBouncer's own connection management.
            from sqlalchemy.pool import NullPool
            _engine = create_async_engine(
                url,
                poolclass=NullPool,
                connect_args=connect_args,
                echo=False,
            )
        else:
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
    Raises HTTP 503 (not 500) for any database unavailability so the frontend
    can display a helpful setup message instead of a generic error.
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
                        "Database unavailable. "
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
                "Cannot connect to database. "
                f"Check DATABASE_URL in Vercel environment variables. ({type(e).__name__})"
            ),
        )
    except Exception as e:
        # Catch raw OS/socket errors (e.g. ConnectionRefusedError) that asyncpg
        # raises before SQLAlchemy can wrap them in OperationalError.
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach database host. "
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
