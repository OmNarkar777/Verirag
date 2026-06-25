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


def get_engine():
    global _engine
    if _engine is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. "
                "Add it in your Vercel project environment variables. "
                "Example: postgresql+asyncpg://user:pass@host:5432/dbname"
            )
        # Use NullPool on Vercel serverless - each function invocation is isolated,
        # so persistent connection pools waste resources and exhaust DB connections.
        if os.environ.get("VERCEL"):
            from sqlalchemy.pool import NullPool
            _engine = create_async_engine(
                settings.database_url,
                poolclass=NullPool,
                echo=False,
            )
        else:
            _engine = create_async_engine(
                settings.database_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
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
    Raises HTTP 503 with a helpful message if DATABASE_URL is not configured.
    """
    from fastapi import HTTPException
    try:
        factory = _get_session_factory()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


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
