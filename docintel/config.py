"""Typed configuration. Every setting comes from the environment (prefix ``DOCINTEL_``) or a ``.env`` file.

There is exactly one definition per setting; modules read ``get_settings()`` and never call ``os.getenv``.
Service addresses and credentials have no defaults: a deployment states them explicitly (``.env.example`` lists
local development values). ``Settings.validate_for_startup`` applies the rules of the environment profile.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]


class ConfigError(RuntimeError):
    """A required setting is missing or a combination of settings is not allowed in this environment."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCINTEL_", env_file=".env", extra="ignore")

    environment: Environment = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # Storage
    database_url: str | None = None
    db_pool_min: int = 2
    db_pool_max: int = 20
    storage_dir: Path = Path("./data/objects")

    # Vector index
    vector_backend: Literal["milvus", "memory", "disabled"] = "milvus"
    milvus_uri: str | None = None
    milvus_token: str | None = None
    milvus_collection_prefix: str = "docintel_chunks"
    milvus_consistency: Literal["Strong", "Bounded", "Session", "Eventually"] = "Bounded"

    # Embeddings
    embedder_url: str | None = None
    embedding_model: str = "all-mpnet-base-v2"
    embed_batch_size: int = 32
    embed_timeout_sec: float = 120.0

    # Optional reranker (served by the embedding service; see docintel/embedder)
    reranker_model: str | None = None
    rerank_top_n: int = 40

    # Optional answer generation (docintel/answering). "extractive" needs no language model.
    answer_provider: Literal["disabled", "extractive", "anthropic"] = "extractive"
    # With "anthropic", the Anthropic SDK resolves credentials from DOCINTEL_ANSWER_API_KEY, else its own environment
    # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile). base_url is for gateways and tests.
    answer_model: str = "claude-opus-5-5"
    answer_api_key: str | None = None
    answer_base_url: str | None = None
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    answer_fallbacks: bool = True          # server-side refusal fallback (Claude API only; disable for gateways)
    answer_timeout_sec: float = 60.0
    answer_max_evidence: int = 12

    # Jobs
    task_mode: Literal["celery", "thread", "inline"] = "thread"
    redis_url: str | None = None
    worker_threads: int = Field(default_factory=lambda: max(2, (os.cpu_count() or 2)))
    ingest_queue: str = "docintel.ingest"
    task_time_limit_sec: int = 1800
    job_heartbeat_sec: int = 30

    # Processing
    date_order: Literal["dmy", "mdy"] = "dmy"    # reading of all-numeric dates like 03/09/2024 (mdy for US documents)
    ocr_languages: str = "eng"
    ocr_dpi: int = 250
    ocr_page_timeout_sec: int = 180              # per page; a page that takes longer fails the document
    signature_detection: bool = True
    ocr_workers: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2)))
    max_upload_bytes: int = 200 * 1024 * 1024
    max_request_bytes: int = 1024 * 1024 * 1024  # all files of one upload request together, checked before spooling
    max_pages: int = 2000
    max_sheet_rows: int = 20000
    max_image_pixels: int = 120_000_000          # decompression-bomb guard for images and rendered pages
    max_archive_uncompressed_bytes: int = 1024 * 1024 * 1024   # OOXML/ODF containers
    max_archive_ratio: int = 200                 # uncompressed / compressed size of a container
    max_attachment_depth: int = 2                # e-mail attachments inside attachments
    soffice_path: str = "soffice"
    convert_timeout_sec: int = 180
    url_fetch_enabled: bool = False
    url_allowed_hosts: str = ""
    url_allow_private: bool = False
    auto_migrate: bool = True

    # Search
    retrieval_engine: Literal["v1", "v2", "shadow"] = "v2"
    result_limit: int = 20
    vector_top_k: int = 100
    lexical_top_k: int = 200
    structured_id_limit: int = 5000
    lexical_avg_unit_terms: float = 120.0        # BM25 length normalization reference
    query_timeout_ms: int = 5000

    # Domain packs enabled when a tenant has no explicit configuration
    default_packs: str = "business,legal,finance"

    # Auth
    auth_mode: Literal["keys", "dev"] = "keys"
    api_keys: str = ""                  # inline JSON (see docintel.security)
    api_keys_file: Path | None = None
    dev_tenant: str = "default"
    cors_origins: str = ""

    # Observability
    metrics_enabled: bool = True
    metrics_token: str | None = None             # when set, /metrics requires "Authorization: Bearer <token>"
    worker_metrics_port: int | None = None       # Celery workers serve their metrics here (internal network only)

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _path(cls, v):
        return Path(v)

    @property
    def is_production(self) -> bool:
        return self.environment in ("production", "staging")

    @property
    def semantic_enabled(self) -> bool:
        return self.vector_backend != "disabled"

    @property
    def packs(self) -> list[str]:
        return [p.strip() for p in self.default_packs.split(",") if p.strip()]

    def require(self, name: str) -> str:
        """The value of a required setting, or a ConfigError naming the environment variable to set."""
        value = getattr(self, name)
        if value in (None, ""):
            raise ConfigError(f"DOCINTEL_{name.upper()} is required but not set")
        return str(value)

    def validate_for_startup(self, role: Literal["api", "worker", "embedder"] = "api") -> None:
        """Fail fast on missing or unsafe configuration for this environment profile."""
        problems = []
        if role in ("api", "worker"):
            for name in ("database_url",):
                if not getattr(self, name):
                    problems.append(f"DOCINTEL_{name.upper()} is required")
            if self.vector_backend == "milvus" and not self.milvus_uri:
                problems.append("DOCINTEL_MILVUS_URI is required with DOCINTEL_VECTOR_BACKEND=milvus")
            if self.semantic_enabled and not self.embedder_url:
                problems.append("DOCINTEL_EMBEDDER_URL is required unless DOCINTEL_VECTOR_BACKEND=disabled")
            if self.task_mode == "celery" and not self.redis_url:
                problems.append("DOCINTEL_REDIS_URL is required with DOCINTEL_TASK_MODE=celery")
            if self.answer_provider == "anthropic":
                try:
                    import anthropic  # noqa: F401
                except ImportError:
                    problems.append("DOCINTEL_ANSWER_PROVIDER=anthropic needs the 'answer' extra (pip install docintel[answer])")
        if self.is_production:
            if self.auth_mode != "keys":
                problems.append(f"DOCINTEL_AUTH_MODE=dev is not allowed in {self.environment}")
            if role == "api" and not (self.api_keys or self.api_keys_file):
                problems.append("API keys (DOCINTEL_API_KEYS_FILE or DOCINTEL_API_KEYS) are required")
            if self.vector_backend == "memory":
                problems.append("DOCINTEL_VECTOR_BACKEND=memory is for tests only")
            if self.task_mode == "inline":
                problems.append("DOCINTEL_TASK_MODE=inline is for tests only")
            for var, model in (("DOCINTEL_EMBEDDING_MODEL", self.embedding_model), ("DOCINTEL_RERANKER_MODEL", self.reranker_model)):
                if model and model.startswith("test-"):
                    problems.append(f"{var}={model} is a stand-in model for tests only")
            if role == "api" and self.metrics_enabled and not self.metrics_token:
                problems.append("DOCINTEL_METRICS_TOKEN is required for /metrics (or set DOCINTEL_METRICS_ENABLED=false)")
            if self.url_fetch_enabled and self.url_allow_private:
                problems.append("DOCINTEL_URL_ALLOW_PRIVATE must be false when URL ingestion is enabled")
        if role in ("api", "worker"):
            from docintel.packs import PackError, get_domain
            try:
                get_domain(tuple(self.packs))              # unknown or malformed packs fail here, not on first use
            except (PackError, OSError, ValueError) as e:
                problems.append(f"DOCINTEL_DEFAULT_PACKS: {e}")
        if problems:
            raise ConfigError("invalid configuration: " + "; ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Forget cached settings (tests)."""
    get_settings.cache_clear()
