from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "LangChain LangGraph App"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"

    # Optional public API protection. If empty, no API key is required.
    api_key: str = ""

    # Keep this app runnable even before Ollama/MCP/database wiring is complete.
    # Set LLM_BACKEND=ollama when your Ollama URL and model are ready.
    llm_backend: Literal["echo", "ollama"] = "echo"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_temperature: float = 0.2
    ollama_timeout_seconds: int = 120

    # Future integrations. Kept here so the deployment contract is ready.
    mcp_server_url: str = "http://host.docker.internal:8002/mcp"
    database_url: str = ""
    redis_url: str = ""

    default_system_prompt: str = Field(
        default=(
            "You are a precise, helpful assistant. Answer directly and be honest "
            "when a tool, database, or model is not configured yet."
        )
    )

    cors_allow_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
