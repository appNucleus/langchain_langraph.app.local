from __future__ import annotations

import ast
from pathlib import Path

from app import __version__
from app.settings import Settings


def test_phase3_runtime_settings_are_available() -> None:
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
    }
    missing = sorted(name for name in required if not hasattr(settings, name))
    assert not missing, f"Missing required Settings attributes: {missing}"


def test_model_role_contract() -> None:
    settings = Settings(_env_file=None)

    assert settings.model_for_key("planner") == settings.model_planner
    assert settings.model_for_key("embedding") == settings.embedding_model
    assert settings.model_for_key("unknown") == settings.model_general
    assert settings.model_role_catalog()["vision"] == settings.model_vision


def test_version_has_one_runtime_source() -> None:
    settings = Settings(_env_file=None)

    assert __version__
    assert "app_version" not in Settings.model_fields
    assert "mcp_client_version" not in Settings.model_fields


def test_all_direct_settings_attributes_exist() -> None:
    """Catch phase-to-phase Settings contract drift before image deployment."""

    settings = Settings(_env_file=None)
    missing: set[str] = set()

    for path in Path("app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if isinstance(node.value, ast.Name) and node.value.id in {
                "settings",
                "app_settings",
                "current",
                "current_settings",
                "self.settings",
            }:
                # self.settings is represented as a nested Attribute, so the
                # direct Name cases are checked here; explicit high-risk
                # contracts are covered by the first test.
                if not hasattr(settings, node.attr):
                    missing.add(f"{path}:{node.attr}")

    assert not missing, "Unknown Settings attributes found:\n" + "\n".join(
        sorted(missing)
    )
