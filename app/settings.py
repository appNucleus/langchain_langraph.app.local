from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
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

    # Run identity. The existing namespace value is intentionally retained for
    # checkpoint compatibility. Change it only with an explicit state migration.
    resume_token_secret: str = ""
    resume_token_ttl_seconds: int = Field(default=3600, ge=60, le=604800)
    run_checkpoint_namespace: str = Field(
        default="phase5-v1",
        min_length=1,
        max_length=100,
    )
    run_state_schema_version: int = Field(default=1, ge=1, le=1000)
    same_conversation_policy: Literal["reject"] = "reject"

    # LLM / Ollama
    llm_backend: Literal["echo", "ollama"] = "echo"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ollama_num_predict: int = Field(default=2048, ge=-1, le=131072)
    # Explicit API context window. Ollama commonly defaults to 4096 tokens on
    # smaller-VRAM systems, which is too small for structured agent prompts that
    # include retrieved evidence and a JSON schema.
    ollama_num_ctx: int = Field(default=8192, ge=2048, le=262144)
    structured_output_reserve_tokens: int = Field(default=1536, ge=256, le=32768)
    structured_prompt_chars_per_token: float = Field(default=3.0, ge=1.0, le=8.0)
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

    # Dependency startup policy. Optional dependencies start in degraded mode;
    # required dependencies fail application startup.
    ollama_required: bool = True
    mcp_required: bool = False

    # Backward-compatible deterministic query and prompt-service settings.
    # These services remain importable and their configuration contract is
    # validated by the CI Settings scanner even when the checkpointed graph is
    # the primary runtime orchestration path.
    default_forecast_days: int = Field(default=7, ge=1, le=14)
    default_news_lookback_days: int = Field(default=7, ge=1, le=90)
    enable_llm_query_planning: bool = False
    max_subqueries: int = Field(default=6, ge=1, le=20)
    max_tool_chars: int = Field(default=12000, ge=1000, le=200000)
    session_history_messages: int = Field(default=12, ge=2, le=200)
    max_conversation_sessions: int = Field(default=1000, ge=1, le=100000)
    conversation_session_ttl_seconds: int = Field(default=3600, ge=60, le=2_592_000)
    conversation_cleanup_interval_seconds: int = Field(default=60, ge=1, le=3600)
    max_history_message_chars: int = Field(default=4000, ge=100, le=100000)

    # Shared capability inventory cache
    inventory_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    inventory_stale_if_error_seconds: int = Field(default=300, ge=0, le=86400)

    # Worker, research, verification, and replanning loop. The former numbered
    # environment names remain accepted through validation aliases.
    agent_max_iterations: int = Field(
        default=4,
        ge=1,
        le=10,
        validation_alias=AliasChoices(
            "AGENT_MAX_ITERATIONS",
            "PHASE2_MAX_ITERATIONS",
            "agent_max_iterations",
            "phase2_max_iterations",
        ),
    )
    agent_max_research_rounds: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices(
            "AGENT_MAX_RESEARCH_ROUNDS",
            "PHASE2_MAX_RESEARCH_ROUNDS",
            "agent_max_research_rounds",
            "phase2_max_research_rounds",
        ),
    )
    agent_max_replans: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias=AliasChoices(
            "AGENT_MAX_REPLANS",
            "PHASE2_MAX_REPLANS",
            "agent_max_replans",
            "phase2_max_replans",
        ),
    )
    agent_max_context_chars: int = Field(
        default=16000,
        ge=2000,
        le=100000,
        validation_alias=AliasChoices(
            "AGENT_MAX_CONTEXT_CHARS",
            "PHASE2_MAX_CONTEXT_CHARS",
            "agent_max_context_chars",
            "phase2_max_context_chars",
        ),
    )
    research_max_queries_per_task: int = Field(default=3, ge=1, le=8)
    research_max_evidence_chars_per_query: int = Field(default=6000, ge=500, le=50000)

    # Global execution budgets
    execution_max_duration_seconds: float = Field(default=180.0, ge=5, le=1800)
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

    # Persistence backends. These remain independent so deployments can use
    # PostgreSQL checkpoints with Redis history, or PostgreSQL for both.
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
    artifact_storage_required: bool = False
    persistence_health_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    # Final-answer verification after multi-task synthesis.
    final_verification_enabled: bool = True
    final_max_revision_rounds: int = Field(default=1, ge=0, le=3)

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        return ["*"] if raw == "*" else [
            item.strip() for item in raw.split(",") if item.strip()
        ]

    @property
    def ollama_max_concurrent_requests(self) -> int:
        """Deprecated compatibility name for ``ollama_max_concurrency``."""

        return self.ollama_max_concurrency

    @property
    def ollama_max_concurrent_heavy_requests(self) -> int:
        """Deprecated compatibility name for ``ollama_heavy_max_concurrency``."""

        return self.ollama_heavy_max_concurrency

    # Compatibility properties for callers that still read the former numbered
    # setting attributes. New code must use the agent_* names above.
    @property
    def phase2_max_iterations(self) -> int:
        return self.agent_max_iterations

    @property
    def phase2_max_research_rounds(self) -> int:
        return self.agent_max_research_rounds

    @property
    def phase2_max_replans(self) -> int:
        return self.agent_max_replans

    @property
    def phase2_max_context_chars(self) -> int:
        return self.agent_max_context_chars

    def model_for_key(self, key: str | None) -> str:
        role = (key or "general").strip().lower()
        return self.model_role_catalog().get(role, self.model_general)

    def model_role_catalog(self) -> dict[str, str]:
        """Return each role exactly once, avoiding silent duplicate-key overrides."""

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
