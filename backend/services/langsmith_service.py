"""services/langsmith_service.py â€” LangSmith integration (gracefully disabled if no key)."""
from __future__ import annotations
from typing import Any
from loguru import logger
from backend.config import get_settings

settings = get_settings()


class LangSmithService:
    def __init__(self):
        self._client = None
        self._enabled = bool(settings.langchain_api_key) and settings.langchain_tracing_v2
        if self._enabled:
            try:
                from langsmith import Client
                self._client = Client(api_key=settings.langchain_api_key)
                logger.info(f"LangSmith connected | project={settings.langchain_project}")
            except Exception as e:
                logger.warning(f"LangSmith init failed (non-fatal): {e}")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def get_project_url(self) -> str | None:
        if not self.enabled:
            return None
        return f"https://smith.langchain.com/projects/{settings.langchain_project}"

    def get_run_url(self, run_id: str) -> str | None:
        if not self.enabled:
            return None
        return f"https://smith.langchain.com/projects/{settings.langchain_project}/runs/{run_id}"

    def list_recent_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.enabled or not self._client:
            return []
        try:
            runs = list(self._client.list_runs(
                project_name=settings.langchain_project, limit=limit, execution_order=1
            ))
            return [
                {
                    "run_id": str(r.id), "name": r.name, "status": r.status,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "url": self.get_run_url(str(r.id)),
                }
                for r in runs
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch LangSmith traces: {e}")
            return []

    def tag_run(self, run_id: str, tags: list[str]) -> bool:
        if not self.enabled or not self._client:
            return False
        try:
            self._client.update_run(run_id=run_id, tags=tags)
            return True
        except Exception as e:
            logger.warning(f"Failed to tag LangSmith run {run_id}: {e}")
            return False


_langsmith_service: LangSmithService | None = None


def get_langsmith_service() -> LangSmithService:
    global _langsmith_service
    if _langsmith_service is None:
        _langsmith_service = LangSmithService()
    return _langsmith_service