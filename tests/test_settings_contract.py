from __future__ import annotations

import ast
from pathlib import Path

from app.settings import Settings


def test_runtime_settings_are_available() -> None:
    settings = Settings(_env_file=None)
    required = {
        "ollama_max_concurrency",
        "ollama_heavy_max_concurrency",
        "ollama_max_concurrent_requests",
        "ollama_max_concurrent_heavy_requests",
        "ollama_num_predict",
        "ollama_think",
        "http_keepalive_expiry_seconds",
        "mcp_read_timeout_seconds",
        "mcp_write_timeout_seconds",
        "inventory_cache_ttl_seconds",
        "inventory_stale_if_error_seconds",
        "agent_max_iterations",
        "agent_max_research_rounds",
        "agent_max_replans",
        "agent_max_context_chars",
    }
    missing = sorted(name for name in required if not hasattr(settings, name))
    assert not missing, f"Missing required Settings attributes: {missing}"


def test_canonical_agent_limit_constructor_names_are_accepted() -> None:
    settings = Settings(
        _env_file=None,
        agent_max_iterations=5,
        agent_max_research_rounds=3,
        agent_max_replans=2,
        agent_max_context_chars=20000,
    )
    assert settings.agent_max_iterations == 5
    assert settings.agent_max_research_rounds == 3
    assert settings.agent_max_replans == 2
    assert settings.agent_max_context_chars == 20000


def test_legacy_agent_limit_constructor_aliases_map_to_canonical_fields() -> None:
    legacy_values = {
        "phase2_max_iterations": 7,
        "phase2_max_research_rounds": 4,
        "phase2_max_replans": 2,
        "phase2_max_context_chars": 24000,
    }
    settings = Settings(_env_file=None, **legacy_values)

    assert settings.agent_max_iterations == 7
    assert settings.agent_max_research_rounds == 4
    assert settings.agent_max_replans == 2
    assert settings.agent_max_context_chars == 24000
    for legacy_name, canonical_name in {
        "phase2_max_iterations": "agent_max_iterations",
        "phase2_max_research_rounds": "agent_max_research_rounds",
        "phase2_max_replans": "agent_max_replans",
        "phase2_max_context_chars": "agent_max_context_chars",
    }.items():
        assert getattr(settings, legacy_name) == getattr(settings, canonical_name)


def test_canonical_agent_limit_environment_names_take_precedence(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "6")
    monkeypatch.setenv("PHASE2_MAX_ITERATIONS", "3")
    settings = Settings(_env_file=None)
    assert settings.agent_max_iterations == 6


def test_legacy_agent_limit_environment_names_remain_accepted(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MAX_RESEARCH_ROUNDS", raising=False)
    monkeypatch.setenv("PHASE2_MAX_RESEARCH_ROUNDS", "3")
    settings = Settings(_env_file=None)
    assert settings.agent_max_research_rounds == 3


def test_model_role_contract() -> None:
    settings = Settings(_env_file=None)
    assert settings.model_for_key("planner") == settings.model_planner
    assert settings.model_for_key("embedding") == settings.embedding_model
    assert settings.model_for_key("unknown") == settings.model_general
    assert settings.model_role_catalog()["vision"] == settings.model_vision


def _is_self_settings(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "settings"
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
    )


def test_direct_self_settings_attributes_exist() -> None:
    """Check unambiguous ``self.settings.<field>`` references only."""

    settings = Settings(_env_file=None)
    missing: set[str] = set()

    for path in Path("app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not _is_self_settings(node.value):
                continue
            if not hasattr(settings, node.attr):
                missing.add(f"{path}:{node.attr}")

    assert not missing, "Unknown Settings attributes found:\n" + "\n".join(
        sorted(missing)
    )
