"""alembic/env.py — Async Alembic migration environment."""
import asyncio
import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv

load_dotenv()

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from backend.database import Base          # noqa: E402
from backend.models import EvalRun, EvalCase, PipelineDocument  # noqa: E402, F401

target_metadata = Base.metadata

_CLOUD_HOSTS = ("supabase.com", "neon.tech", "render.com")


def get_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql+asyncpg://verirag:verirag_secret@localhost:5432/verirag")
    if not url.startswith("postgresql+asyncpg"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Remove psycopg2-style sslmode param that asyncpg ignores/warns about
    url = re.sub(r"[?&]sslmode=\w+", "", url)
    return url


def get_connect_args(url: str) -> dict:
    if any(h in url for h in _CLOUD_HOSTS):
        return {"ssl": "require"}
    return {}


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(), target_metadata=target_metadata,
        literal_binds=True, dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      include_schemas=True, compare_server_default=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    url = get_url()
    connectable = create_async_engine(url, connect_args=get_connect_args(url))
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
