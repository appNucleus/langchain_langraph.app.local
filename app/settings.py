from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "LangChain LangGraph App"
    environment: str = "local"
    log_level: str = "INFO"
    api_key: str = ""
    cors_allow_origins: str = "*"

    llm_backend: Literal["echo", "ollama"] = "echo"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ollama_timeout_seconds: float = Field(default=180, gt=0)
    ollama_connect_timeout_seconds: float = Field(default=10, gt=0)
    ollama_pool_timeout_seconds: float = Field(default=30, gt=0)
    ollama_max_connections: int = Field(default=20, ge=1, le=200)
    ollama_max_keepalive_connections: int = Field(default=10, ge=0, le=200)
    ollama_max_concurrency: int = Field(default=2, ge=1, le=16)
    ollama_heavy_max_concurrency: int = Field(default=1, ge=1, le=4)
    ollama_keep_alive: str = "10m"
    ollama_num_predict: int = Field(default=2048, ge=-1, le=32768)
    ollama_think: bool = False
    http_keepalive_expiry_seconds: float = Field(default=30, gt=0, le=600)

    model_planner: str = "qwen3.5:4b"
    model_simple: str = "qwen3.5:2b"
    model_general: str = "qwen3.5:4b"
    model_search: str = "qwen3.5:9b"
    model_reasoning: str = "deepseek-r1:8b"
    model_fast_reasoning: str = "phi4-mini-reasoning:latest"
    model_heavy: str = "gemma4:26b-a4b-it-qat"
    model_synthesis: str = "gemma4:12b-it-qat"
    model_writer: str = "gemma4:e4b-it-qat"
    model_classifier: str = "gemma4:e2b-it-qat"
    model_vision: str = "qwen3-vl:4b"
    embedding_model: str = "qwen3-embedding:0.6b"
    model_fallback: str = "granite3.3:8b"

    mcp_server_url: str = "http://host.docker.internal:8002/mcp"
    mcp_enabled: bool = False
    mcp_verify_tls: bool = False
    mcp_follow_redirects: bool = True
    mcp_timeout_seconds: float = Field(default=60, gt=0)
    mcp_connect_timeout_seconds: float = Field(default=10, gt=0)
    mcp_pool_timeout_seconds: float = Field(default=20, gt=0)
    mcp_max_connections: int = Field(default=30, ge=1, le=300)
    mcp_max_keepalive_connections: int = Field(default=15, ge=0, le=300)
    mcp_max_concurrency: int = Field(default=6, ge=1, le=32)
    mcp_session_enabled: bool = True
    mcp_initialize_on_startup: bool = True
    mcp_protocol_version: str = "2025-06-18"
    mcp_client_name: str = "langchain-langraph-app"

    inventory_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    inventory_stale_if_error_seconds: int = Field(default=300, ge=0, le=86400)

    phase2_max_iterations: int = Field(default=4, ge=1, le=10)
    phase2_max_research_rounds: int = Field(default=2, ge=0, le=5)
    phase2_max_replans: int = Field(default=1, ge=0, le=3)
    phase2_max_context_chars: int = Field(default=16000, ge=2000, le=100000)

    execution_max_duration_seconds: float = Field(default=180, ge=5, le=1800)
    execution_max_model_calls: int = Field(default=8, ge=1, le=50)
    execution_max_tool_calls: int = Field(default=10, ge=0, le=100)
    execution_max_verifier_rounds: int = Field(default=3, ge=1, le=10)

    state_ttl_seconds: int = Field(default=3600, ge=60, le=2_592_000)
    state_max_sessions: int = Field(default=1000, ge=10, le=100000)
    state_max_history_messages: int = Field(default=30, ge=2, le=500)
    side_effect_policy_enabled: bool = True
    real_streaming_enabled: bool = True
    detailed_tracing_enabled: bool = True
    expose_internal_health_details: bool = False

    # Phase 4: backends are independent so deployments can use PostgreSQL
    # checkpoints with Redis history/cache or PostgreSQL for both.
    state_backend: Literal["memory", "redis", "postgres"] = "memory"
    checkpoint_backend: Literal["memory", "postgres"] = "memory"
    artifact_backend: Literal["disabled", "minio"] = "disabled"

    database_url: str = ""
    postgres_pool_min_size: int = Field(default=1, ge=1, le=20)
    postgres_pool_max_size: int = Field(default=10, ge=1, le=100)
    postgres_command_timeout_seconds: float = Field(default=30, gt=0, le=300)
    postgres_auto_setup: bool = True

    redis_url: str = ""
    redis_key_prefix: str = "langgraph"

    minio_endpoint: str = "dbs.home.arpa:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "langchain-langraph-app"
    minio_secure: bool = False

    persistence_required: bool = False

    default_system_prompt: str = (
        "You are a precise, evidence-grounded assistant. "
        "Be honest about uncertainty and missing tools."
    )

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    def model_for_key(self, key: str | None) -> str:
        role = (key or "general").strip().lower()
        return self.model_role_catalog().get(role, self.model_general)

    def model_role_catalog(self) -> dict[str, str]:
        return {
            "planner": self.model_planner,
            "simple": self.model_simple,
            "general": self.model_general,
            "search": self.model_search,
            "reasoning": self.model_reasoning,
            "fast_reasoning": self.model_fast_reasoning,
            "heavy": self.model_heavy,
            "synthesis": self.model_synthesis,
            "writer": self.model_writer,
            "classifier": self.model_classifier,
            "vision": self.model_vision,
            "embedding": self.embedding_model,
            "fallback": self.model_fallback,
        }

    @property
    def persistence_enabled(self) -> bool:
        return any(
            (
                self.state_backend != "memory",
                self.checkpoint_backend != "memory",
                self.artifact_backend != "disabled",
            )
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
