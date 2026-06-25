"""config.py - Centralized settings via pydantic-settings."""
from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM - empty default so import succeeds; endpoints degrade gracefully when unset
    groq_api_key: str = Field(default="", description="Groq API key (required for LLM features)")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.0)

    # Observability - optional
    langchain_api_key: str = Field(default="", description="LangSmith API key (optional)")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="verirag-prod")

    # Database - empty default so import succeeds; engine created lazily at first use
    database_url: str = Field(default="", description="postgresql+asyncpg:// connection URL")

    # ChromaDB - /tmp is writable on Vercel serverless; override via CHROMA_PERSIST_DIR env var
    chroma_persist_dir: str = Field(default="/tmp/chroma_data")
    chroma_collection_name: str = Field(default="verirag_docs")

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    hf_token: str = Field(default="", description="HuggingFace token for Inference API embeddings")

    # RAG
    retrieval_top_k: int = Field(default=5)
    retrieval_lambda: float = Field(default=0.5)

    # Regression detection - 0.10 = 10-point absolute drop triggers flag
    regression_threshold: float = Field(default=0.10)

    # Rate limiting - each RAGAS run makes ~200 Groq calls; cap concurrency
    max_concurrent_evals: int = Field(default=5)

    # CORS - space or comma-separated list of allowed origins; * for open access
    cors_origins: str = Field(default="*", description="Allowed CORS origins (comma-separated)")

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_v1_prefix: str = Field(default="/api/v1")

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field
    @property
    def allowed_origins(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.replace(",", " ").split() if o.strip()]

    @computed_field
    @property
    def is_configured(self) -> bool:
        """True when all required services are configured."""
        return bool(self.database_url and self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
