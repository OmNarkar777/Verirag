"""config.py â€” Centralized settings via pydantic-settings."""
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

    # LLM
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.0)

    # Observability â€” optional so app starts without LangSmith
    langchain_api_key: str = Field(default="", description="LangSmith API key (optional)")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="verirag-prod")

    # Database
    database_url: str = Field(..., description="postgresql+asyncpg:// required")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="verirag_docs")

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    hf_token: str = Field(default="", description="HuggingFace token for Inference API embeddings")

    # RAG
    retrieval_top_k: int = Field(default=5)
    retrieval_lambda: float = Field(default=0.5)

    # Regression detection â€” 0.10 = 10-point absolute drop triggers flag
    regression_threshold: float = Field(default=0.10)

    # Rate limiting â€” each RAGAS run makes ~200 Groq calls; cap concurrency
    max_concurrent_evals: int = Field(default=5)

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1")

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()