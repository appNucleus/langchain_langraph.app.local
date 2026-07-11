from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration shared by Phases 1, 2, and 3.

    Environment variable names remain compatible with the existing runtime.env.
    Application versioning intentionally does not belong here; use app.__version__.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "LangChain LangGraph App"
    environment: str = "local"
    log_level: str = "INFO"
    api_key: str = ""
    cors_allow_origins: str = "*"
    default_system_prompt: str = (
        "You are a precise, evidence-grounded assistant. "
        "Be honest about uncertainty and missing tools."
    )

    # LLM / Ollama
    llm_backend: Literal["echo", "ollama"] = "echo"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ollama_num_predict: int = Field(default=2048, ge=-1, le=131072)
    ollama_think: bool = False
    ollama_timeout_seconds: float = Field(default=180.0, gt=0)
    ollama_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    ollama_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    ollama_max_connections: int = Field(default=20, ge=1, le=1000)
    ollama_max_keepalive_connections: int = Field(default=10, ge=0, le=1000)
    ollama_max_concurrency: int = Field(default=2, ge=1, le=16)
    ollama_heavy_max_concurrency: int = Field(default=1, ge=1, le=4)
    ollama_keep_alive: str = "10m"
    http_keepalive_expiry_seconds: float = Field(default=30.0, gt=0)

    # Model role assignments
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
    model_fallback: str = "granite3.3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"

    # MCP
    mcp_server_url: str = "http://host.docker.internal:8002/mcp"
    mcp_enabled: bool = False
    mcp_verify_tls: bool = False
    mcp_follow_redirects: bool = True
    mcp_timeout_seconds: float = Field(default=60.0, gt=0)
    mcp_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    mcp_read_timeout_seconds: float = Field(default=60.0, gt=0)
    mcp_write_timeout_seconds: float = Field(default=60.0, gt=0)
    mcp_pool_timeout_seconds: float = Field(default=20.0, gt=0)
    mcp_max_connections: int = Field(default=30, ge=1, le=1000)
    mcp_max_keepalive_connections: int = Field(default=15, ge=0, le=1000)
    mcp_max_concurrency: int = Field(default=6, ge=1, le=32)
    mcp_session_enabled: bool = True
    mcp_initialize_on_startup: bool = True
    mcp_protocol_version: str = "2025-06-18"
    mcp_client_name: str = "langchain-langraph-app"
    mcp_inventory_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)

    # Unified runtime inventory cache used by app.services.inventory
    inventory_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    inventory_stale_if_error_seconds: int = Field(default=300, ge=0, le=86400)

    # Phase 2 agent loop
    phase2_max_iterations: int = Field(default=4, ge=1, le=10)
    phase2_max_research_rounds: int = Field(default=2, ge=0, le=5)
    phase2_max_replans: int = Field(default=1, ge=0, le=3)
    phase2_max_context_chars: int = Field(default=16000, ge=2000, le=100000)

    # Phase 3 global execution budgets
    execution_max_duration_seconds: float = Field(default=180.0, ge=5, le=1800)
    execution_max_model_calls: int = Field(default=8, ge=1, le=50)
    execution_max_tool_calls: int = Field(default=10, ge=0, le=100)
    execution_max_verifier_rounds: int = Field(default=3, ge=1, le=10)

    # Phase 3 bounded in-memory state
    state_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    state_max_sessions: int = Field(default=1000, ge=10, le=100000)
    state_max_history_messages: int = Field(default=30, ge=2, le=200)

    # Phase 3 safety / telemetry
    side_effect_policy_enabled: bool = True
    real_streaming_enabled: bool = True
    detailed_tracing_enabled: bool = True
    expose_internal_health_details: bool = False

    # Phase 4 placeholders; unused unless a persistence backend is enabled.
    database_url: str = ""
    redis_url: str = ""

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        return ["*"] if raw == "*" else [
            item.strip() for item in raw.split(",") if item.strip()
        ]

    @property
    def ollama_max_concurrent_requests(self) -> int:
        """Backward-compatible Phase 1 name used by older resource managers."""
        return self.ollama_max_concurrency

    @property
    def ollama_max_concurrent_heavy_requests(self) -> int:
        """Backward-compatible Phase 1 name used by older resource managers."""
        return self.ollama_heavy_max_concurrency

    def model_for_key(self, key: str | None) -> str:
        role = (key or "general").strip().lower()
        mapping = {
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
            "fallback": self.model_fallback,
            "embedding": self.embedding_model,
        }
        return mapping.get(role, self.model_general)

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
            "fallback": self.model_fallback,
            "embedding": self.embedding_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
