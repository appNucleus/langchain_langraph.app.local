from __future__ import annotations

import pytest

from app.settings import Settings
from app.state.run_repository import MemoryRunRepository, PostgresRunRepository
from app.state.runtime import StateRuntime


def _settings(**overrides: object) -> Settings:
    values = {
        "_env_file": None,
        "llm_backend": "echo",
        "ollama_required": False,
        "mcp_required": False,
        "state_backend": "memory",
        "run_repository_backend": "memory",
        "checkpoint_backend": "memory",
        "artifact_backend": "disabled",
        "resume_token_secret": "test-secret",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_state_runtime_starts_memory_run_repository() -> None:
    runtime = StateRuntime(_settings())
    await runtime.start()
    try:
        assert isinstance(runtime.runs, MemoryRunRepository)
        health = await runtime.health()
        assert health["runs"]["status"] == "available"
        assert health["runs"]["backend"] == "memory"
    finally:
        await runtime.aclose()


def test_postgres_run_repository_requires_database_url() -> None:
    runtime = StateRuntime(
        _settings(run_repository_backend="postgres", database_url="")
    )
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        runtime._build_run_repository()


def test_postgres_run_repository_is_selected_explicitly() -> None:
    runtime = StateRuntime(
        _settings(
            run_repository_backend="postgres",
            database_url="postgresql://example.invalid/app",
        )
    )
    repository = runtime._build_run_repository()
    assert isinstance(repository, PostgresRunRepository)


def test_postgres_run_repository_requires_persistent_token_key() -> None:
    with pytest.raises(ValueError, match="persistent resume-token signing key"):
        Settings(
            _env_file=None,
            llm_backend="echo",
            ollama_required=False,
            mcp_required=False,
            run_repository_backend="postgres",
            database_url="postgresql://example.invalid/app",
            resume_token_secret="",
            resume_token_keys_json="",
            api_key="",
        )
