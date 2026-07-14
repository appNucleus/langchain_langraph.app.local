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
