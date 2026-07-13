from __future__ import annotations

import re

from app import __version__
from app.observability import InMemoryMetrics, MetricsRegistry
from app.settings import Settings


_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def test_version_has_single_runtime_source() -> None:
    assert _SEMVER.fullmatch(__version__)
    assert "app_version" not in Settings.model_fields
    assert "mcp_client_version" not in Settings.model_fields


def test_metrics_backward_compatibility_and_bounds() -> None:
    registry = InMemoryMetrics(max_timing_samples=2)
    assert isinstance(registry, MetricsRegistry)

    registry.inc("requests")
    registry.observe("latency", 1.0)
    registry.observe("latency", 2.0)
    registry.observe("latency", 3.0)
    snapshot = registry.snapshot()

    assert snapshot["counters"]["requests"] == 1
    assert snapshot["timings"]["latency"] == {
        "count": 2,
        "avg": 2.5,
        "min": 2.0,
        "max": 3.0,
    }


def test_runtime_settings_cover_clients_and_inventory() -> None:
    settings = Settings(
        llm_backend="echo",
        mcp_enabled=False,
        _env_file=None,
    )
    assert settings.http_keepalive_expiry_seconds > 0
    assert settings.ollama_num_predict >= -1
    assert settings.inventory_cache_ttl_seconds > 0
    assert settings.inventory_stale_if_error_seconds >= 0
    assert settings.model_for_key("general") == settings.model_general
    assert settings.model_role_catalog()["embedding"] == settings.embedding_model
