from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env.

    Defaults are tuned for the user's trusted local network:
    - Ollama on http://ollama.home.arpa:11434
    - MCP through stable network DNS: https://mcp.home.arpa/mcp

    Model defaults intentionally map every installed local model to at least one
    explicit task family. The router then resolves these roles against the live
    Ollama inventory so unavailable models are not selected at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "LangChain LangGraph App"
    app_version: str = "0.6.0-phase1"
    environment: str = "local"
    log_level: str = "INFO"

    # Optional API protection. If empty, no API key is required.
    api_key: str = ""

    # LLM backend. Keep echo available for smoke tests and no-LLM fallback.
    llm_backend: Literal["echo", "ollama"] = "ollama"

    # Ollama server.
    ollama_base_url: str = "http://ollama.home.arpa:11434"
    ollama_timeout_seconds: float = 240.0
    ollama_connect_timeout_seconds: float = 10.0
    ollama_pool_timeout_seconds: float = 30.0
    ollama_max_connections: int = 20
    ollama_max_keepalive_connections: int = 10
    ollama_max_concurrent_requests: int = 2
    ollama_max_concurrent_heavy_requests: int = 1
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 2048
    ollama_stream_num_predict: int = 2048
    ollama_keep_alive: str = "10m"
    ollama_think: bool = False

    # Model router defaults based on the user's installed Ollama models.
    # Core pattern requested by the user:
    # simple/general -> qwen3.5:2b / qwen3.5:4b
    # search-heavy   -> qwen3.5:9b
    # reasoning      -> deepseek-r1:8b / phi4-mini-reasoning
    # synthesis      -> gemma4:12b-it-qat / gemma4:26b-a4b-it-qat
    # vision         -> qwen3-vl:4b
    # embedding      -> qwen3-embedding:0.6b
    # fallback       -> granite3.3:8b
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

    # MCP server. Use DNS/Caddy/network routing, not Docker-internal host aliases.
    mcp_enabled: bool = True
    mcp_server_url: str = "https://mcp.home.arpa/mcp"
    # mcp_timeout_seconds remains as a backward-compatible umbrella value.
    mcp_timeout_seconds: float = 75.0
    mcp_connect_timeout_seconds: float = 10.0
    mcp_read_timeout_seconds: float = 75.0
    mcp_write_timeout_seconds: float = 30.0
    mcp_pool_timeout_seconds: float = 30.0
    mcp_max_connections: int = 20
    mcp_max_keepalive_connections: int = 10
    mcp_session_enabled: bool = True
    mcp_initialize_on_startup: bool = False
    mcp_client_name: str = "app.local"
    mcp_client_version: str = "0.6.0-phase1"
    mcp_verify_tls: bool = False
    mcp_follow_redirects: bool = True
    mcp_protocol_version: str = "2025-03-26"

    # Tool/query behavior.
    default_search_results: int = 8
    default_scrape_pages: int = 4
    max_tool_calls: int = 4
    max_tool_chars: int = 18000
    max_references: int = 8
    answer_reference_limit: int = 3
    max_subqueries: int = 10
    enable_llm_query_planning: bool = True
    search_unknown_or_fresh: bool = True
    default_news_lookback_days: int = 7
    default_forecast_days: int = 7

    # Short-lived runtime inventory cache. Set to 0 to disable caching.
    inventory_cache_ttl_seconds: float = 30.0
    inventory_stale_if_error_seconds: float = 300.0

    # Shared HTTP connection lifetime.
    http_keepalive_expiry_seconds: float = 30.0

    # Sub-answer validation/retry behavior. This blocks broken outputs like "**" or "1".
    min_answer_chars: int = 80
    max_answer_retries: int = 2

    # In-memory session history. This is intentionally simple and local.
    session_history_messages: int = 12
    max_history_message_chars: int = 1600
    max_conversation_sessions: int = 500
    conversation_session_ttl_seconds: float = 21600.0
    conversation_cleanup_interval_seconds: float = 300.0

    default_system_prompt: str = Field(
        default=(
            "You are a precise local assistant running behind a FastAPI + LangGraph service. "
            "Use tool evidence when provided. Be direct, technically accurate, and honest about uncertainty. "
            "Never expose hidden reasoning or thinking traces. Prefer broad, useful answers over narrow one-line replies. "
            "Use very simple wording unless the user asks for deep technical detail. When references are available, ground the answer in them. "
            "Prefer official documentation and high-quality primary sources over blogs, forums, Reddit, or SEO pages."
        )
    )

    cors_allow_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    def model_for_key(self, key: str | None) -> str:
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
        return mapping.get(key or "general", self.model_fallback)

    def model_role_catalog(self) -> list[dict[str, str]]:
        """Human/readable inventory role map; every installed model has a task family."""
        return [
            {"role": "simple", "model": self.model_simple, "task": "Very short/simple answers, quick summaries, tiny transformations."},
            {"role": "general", "model": self.model_general, "task": "Normal chat, query planning, broad explanations, and app-control questions."},
            {"role": "search", "model": self.model_search, "task": "Search-heavy/current/fact-checking answers using MCP web/news tools."},
            {"role": "reasoning", "model": self.model_reasoning, "task": "Deep reasoning, architecture tradeoffs, debugging, and multi-step analysis."},
            {"role": "fast_reasoning", "model": self.model_fast_reasoning, "task": "Faster math/logic/algorithm checks and retry path for reasoning answers."},
            {"role": "synthesis", "model": self.model_synthesis, "task": "Combine multiple subquery answers into one clean final answer."},
            {"role": "heavy", "model": self.model_heavy, "task": "Highest-quality final synthesis or very complex long-context work."},
            {"role": "writer", "model": self.model_writer, "task": "Long-form rewriting, polished documentation, professional email/report drafts."},
            {"role": "classifier", "model": self.model_classifier, "task": "Intent classification, routing checks, and compact yes/no/schema decisions."},
            {"role": "vision", "model": self.model_vision, "task": "Image, screenshot, chart, or visual-question understanding."},
            {"role": "embedding", "model": self.embedding_model, "task": "Vector embedding for semantic search/RAG indexing; not used as a chat model."},
            {"role": "fallback", "model": self.model_fallback, "task": "Safe fallback when preferred role models are unavailable."},
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
