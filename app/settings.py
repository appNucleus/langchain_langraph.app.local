from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', case_sensitive=False)
    app_name: str = 'LangChain LangGraph App'
    app_version: str = '0.3.0'
    environment: str = 'local'
    log_level: str = 'INFO'
    api_key: str = ''
    llm_backend: Literal['echo','ollama'] = 'echo'
    ollama_base_url: str = 'http://host.docker.internal:11434'
    ollama_model: str = 'qwen3.5:4b'
    ollama_temperature: float = 0.2
    ollama_timeout_seconds: float = 180
    ollama_connect_timeout_seconds: float = 10
    ollama_pool_timeout_seconds: float = 30
    ollama_max_connections: int = 20
    ollama_max_keepalive_connections: int = 10
    ollama_max_concurrency: int = Field(default=2, ge=1, le=16)
    ollama_heavy_max_concurrency: int = Field(default=1, ge=1, le=4)
    ollama_keep_alive: str = '10m'
    model_planner: str = 'qwen3.5:4b'
    model_general: str = 'qwen3.5:4b'
    model_reasoning: str = 'deepseek-r1:8b'
    model_synthesis: str = 'gemma4:12b-it-qat'
    model_simple: str = 'qwen3.5:2b'
    model_search: str = 'qwen3.5:9b'
    model_fast_reasoning: str = 'phi4-mini-reasoning:latest'
    model_heavy: str = 'gemma4:26b-a4b-it-qat'
    model_writer: str = 'gemma4:e4b-it-qat'
    model_classifier: str = 'gemma4:e2b-it-qat'
    model_vision: str = 'qwen3-vl:4b'
    embedding_model: str = 'qwen3-embedding:0.6b'
    model_fallback: str = 'granite3.3:8b'
    mcp_server_url: str = 'http://host.docker.internal:8002/mcp'
    mcp_enabled: bool = False
    mcp_verify_tls: bool = False
    mcp_follow_redirects: bool = True
    mcp_timeout_seconds: float = 60
    mcp_connect_timeout_seconds: float = 10
    mcp_pool_timeout_seconds: float = 20
    mcp_max_connections: int = 30
    mcp_max_keepalive_connections: int = 15
    mcp_max_concurrency: int = Field(default=6, ge=1, le=32)
    mcp_session_enabled: bool = True
    mcp_initialize_on_startup: bool = True
    mcp_protocol_version: str = '2025-06-18'
    mcp_client_name: str = 'langchain-langraph-app'
    mcp_client_version: str = ''
    mcp_inventory_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    database_url: str = ''
    redis_url: str = ''
    phase2_max_iterations: int = Field(default=4, ge=1, le=10)
    phase2_max_research_rounds: int = Field(default=2, ge=0, le=5)
    phase2_max_replans: int = Field(default=1, ge=0, le=3)
    phase2_max_context_chars: int = Field(default=16000, ge=2000, le=100000)
    execution_max_duration_seconds: float = Field(default=180, ge=5, le=1800)
    execution_max_model_calls: int = Field(default=8, ge=1, le=50)
    execution_max_tool_calls: int = Field(default=10, ge=0, le=100)
    execution_max_verifier_rounds: int = Field(default=3, ge=1, le=10)
    state_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    state_max_sessions: int = Field(default=1000, ge=10, le=100000)
    state_max_history_messages: int = Field(default=30, ge=2, le=200)
    side_effect_policy_enabled: bool = True
    real_streaming_enabled: bool = True
    detailed_tracing_enabled: bool = True
    expose_internal_health_details: bool = False
    default_system_prompt: str = 'You are a precise, evidence-grounded assistant. Be honest about uncertainty and missing tools.'
    cors_allow_origins: str = '*'
    @property
    def cors_origins(self) -> list[str]:
        raw=self.cors_allow_origins.strip()
        return ['*'] if raw=='*' else [x.strip() for x in raw.split(',') if x.strip()]

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
