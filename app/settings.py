from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env.

    The defaults are tuned for the user's trusted local network:
    - Ollama on http://ollama.home.arpa:11434
    - MCP on https://mcp.home.arpa/mcp with local/internal TLS
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "LangChain LangGraph App"
    app_version: str = "0.3.0"
    environment: str = "local"
    log_level: str = "INFO"

    # Optional API protection. If empty, no API key is required.
    api_key: str = ""

    # LLM backend. Keep echo available for smoke tests and no-LLM fallback.
    llm_backend: Literal["echo", "ollama"] = "ollama"

    # Ollama server.
    ollama_base_url: str = "http://ollama.home.arpa:11434"
    ollama_timeout_seconds: float = 240.0
    ollama_temperature: float = 0.1
    ollama_num_predict: int = 2048
    ollama_stream_num_predict: int = 2048
    ollama_keep_alive: str = "10m"
    ollama_think: bool = False

    # Model router defaults based on the user's installed Ollama models.
    model_simple: str = "qwen3.5:2b"
    model_general: str = "qwen3.5:4b"
    model_search: str = "qwen3.5:9b"
    model_reasoning: str = "phi4-mini-reasoning:latest"
    model_heavy: str = "gemma4:12b-it-qat"
    model_vision: str = "qwen3-vl:4b"
    model_fallback: str = "qwen3.5:4b"
    embedding_model: str = "qwen3-embedding:0.6b"

    # MCP server.
    mcp_enabled: bool = True
    mcp_server_url: str = "https://mcp.home.arpa/mcp"
    mcp_timeout_seconds: float = 75.0
    mcp_verify_tls: bool = False
    mcp_protocol_version: str = "2025-03-26"

    # Tool/query behavior.
    default_search_results: int = 8
    default_scrape_pages: int = 4
    max_tool_calls: int = 4
    max_tool_chars: int = 14000
    max_references: int = 8
    default_news_lookback_days: int = 7
    default_forecast_days: int = 7

    # In-memory session history. This is intentionally simple and local.
    session_history_messages: int = 12
    max_history_message_chars: int = 1600

    default_system_prompt: str = Field(
        default=(
            "You are a precise local assistant running behind a FastAPI + LangGraph service. "
            "Use tool evidence when provided. Be direct, technically accurate, and honest about uncertainty. "
            "Never expose hidden reasoning or thinking traces. When references are available, ground the answer in them."
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
            "simple": self.model_simple,
            "general": self.model_general,
            "search": self.model_search,
            "reasoning": self.model_reasoning,
            "heavy": self.model_heavy,
            "vision": self.model_vision,
            "fallback": self.model_fallback,
        }
        return mapping.get(key or "general", self.model_fallback)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
