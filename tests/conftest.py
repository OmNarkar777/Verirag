"""
tests/conftest.py - Shared pytest configuration and fixtures.

Sets asyncio_mode="auto" so all async test functions run without
needing @pytest.mark.asyncio on each one individually.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

pytest_plugins = ("anyio",)


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")


# ---------------------------------------------------------------------------
# Async session fixture — shared DB override for endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_session():
    """A MagicMock that satisfies AsyncSession's interface."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.delete = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def override_get_db(mock_db_session):
    """
    FastAPI dependency override that replaces get_db with an in-memory mock.
    Usage:
        def test_foo(override_get_db):
            ...  # get_db is overridden for the duration of this test
    """
    from backend.main import app
    from backend.database import get_db

    async def _mock_db():
        yield mock_db_session

    app.dependency_overrides[get_db] = _mock_db
    yield mock_db_session
    app.dependency_overrides.pop(get_db, None)
