from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', case_sensitive=False)
    app_name: str = 'LangChain LangGraph App'
    app_version: str = '0.2.0'
    environment: str = 'local'
    log_level: str = 'INFO'
    api_key: str = ''
    llm_backend: Literal['echo','ollama'] = 'echo'
    ollama_base_url: str = 'http://host.docker.internal:11434'
    ollama_model: str = 'qwen2.5:0.5b'
    ollama_temperature: float = 0.2
    ollama_timeout_seconds: int = 120
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
    mcp_timeout_seconds: int = 60
    database_url: str = ''
    redis_url: str = ''
    phase2_max_iterations: int = Field(default=4, ge=1, le=10)
    phase2_max_research_rounds: int = Field(default=2, ge=0, le=5)
    phase2_max_replans: int = Field(default=1, ge=0, le=3)
    phase2_max_context_chars: int = Field(default=16000, ge=2000, le=100000)
    default_system_prompt: str = 'You are a precise, evidence-grounded assistant. Be honest about uncertainty and missing tools.'
    cors_allow_origins: str = '*'
    @property
    def cors_origins(self) -> list[str]:
        raw=self.cors_allow_origins.strip(); return ['*'] if raw=='*' else [x.strip() for x in raw.split(',') if x.strip()]

@lru_cache(maxsize=1)
def get_settings() -> Settings: return Settings()
