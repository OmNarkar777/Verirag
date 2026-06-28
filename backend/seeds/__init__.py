"""
seeds/__init__.py — Automatic database seeding.

Seeds realistic eval data and demo documents when the database is empty.
Runs at startup via the lifespan hook. Idempotent — skips when data exists.
"""
from __future__ import annotations

import os
from loguru import logger


def _purge_duplicate_sample_runs() -> None:
    """
    Delete bulk-duplicate sample/test eval runs created during development testing.
    Runs where the same version_tag appears >3 times are almost certainly
    test artifacts (e.g. polling loops). Keeps 3 most-recent per duplicate tag.
    Also removes known one-off test tags (v0.0.x-sync, v0.0.x-verify, etc.).
    """
    from sqlalchemy import create_engine, select, func, delete as sa_delete
    from sqlalchemy.orm import sessionmaker
    from backend.database import _get_sync_engine
    from backend.models import EvalRun

    engine = _get_sync_engine()
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    with Session() as sess:
        # Find version tags with more than 3 copies
        counts = sess.execute(
            select(EvalRun.version_tag, func.count(EvalRun.id).label("cnt"))
            .group_by(EvalRun.version_tag)
            .having(func.count(EvalRun.id) > 3)
        ).all()

        total_deleted = 0
        for tag, cnt in counts:
            # Keep the 3 most recent, delete the rest
            keep_ids = [
                row.id for row in sess.execute(
                    select(EvalRun.id)
                    .where(EvalRun.version_tag == tag)
                    .order_by(EvalRun.created_at.desc())
                    .limit(3)
                ).all()
            ]
            result = sess.execute(
                sa_delete(EvalRun).where(
                    EvalRun.version_tag == tag,
                    EvalRun.id.not_in(keep_ids),
                )
            )
            deleted = result.rowcount
            total_deleted += deleted
            if deleted:
                logger.info(f"Purged {deleted} duplicate runs for version_tag={tag!r}")

        if total_deleted:
            sess.commit()
            logger.info(f"Cleanup complete — removed {total_deleted} duplicate eval runs")


def seed_all_sync() -> None:
    """
    Seed eval runs + demo documents synchronously.
    Called from lifespan via asyncio.to_thread() so it doesn't block the event loop.
    """
    try:
        _purge_duplicate_sample_runs()
    except Exception as e:
        logger.error(f"Duplicate run cleanup failed (non-fatal): {e}")

    try:
        from backend.seeds.eval_data import seed_eval_data_sync
        seed_eval_data_sync()
    except Exception as e:
        logger.error(f"Eval data seeding failed (non-fatal): {e}")

    try:
        from backend.seeds.documents import seed_demo_documents
        seed_demo_documents()
    except Exception as e:
        logger.error(f"Document seeding failed (non-fatal): {e}")
