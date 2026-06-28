"""
seeds/__init__.py — Automatic database seeding.

Seeds realistic eval data and demo documents when the database is empty.
Runs at startup via the lifespan hook. Idempotent — skips when data exists.
"""
from __future__ import annotations

import os
from loguru import logger


def seed_all_sync() -> None:
    """
    Seed eval runs + demo documents synchronously.
    Called from lifespan via asyncio.to_thread() so it doesn't block the event loop.
    """
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
