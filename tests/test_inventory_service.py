from __future__ import annotations

from typing import Any

import pytest

from app.services.inventory import InventoryService
from app.settings import Settings


class FakeOllama:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    async def list_models(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("ollama unavailable")
        return [{"name": f"model-{self.calls}:4b"}]


class FakeMCP:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    async def list_tools(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("mcp unavailable")
        return [{"name": f"tool-{self.calls}", "inputSchema": {"type": "object"}}]


@pytest.mark.asyncio
async def test_cache_returns_defensive_copies_and_avoids_duplicate_refreshes() -> None:
    settings = Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        inventory_cache_ttl_seconds=60,
    )
    ollama = FakeOllama()
    mcp = FakeMCP()
    service = InventoryService(settings, ollama, mcp)

    first = await service.load()
    first.models[0]["name"] = "mutated"
    second = await service.load()

    assert second.cached is True
    assert second.model_names == ["model-1:4b"]
    assert second.tool_names == ["tool-1"]
    assert ollama.calls == mcp.calls == 1


@pytest.mark.asyncio
async def test_refresh_uses_stale_failed_source_without_discarding_fresh_source() -> None:
    settings = Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        inventory_cache_ttl_seconds=60,
        inventory_stale_if_error_seconds=300,
    )
    ollama = FakeOllama()
    mcp = FakeMCP()
    service = InventoryService(settings, ollama, mcp)

    initial = await service.load(force_refresh=True)
    assert initial.tool_names == ["tool-1"]

    mcp.fail = True
    refreshed = await service.load(force_refresh=True)

    assert refreshed.model_names == ["model-2:4b"]
    assert refreshed.tool_names == ["tool-1"]
    assert refreshed.cached is True
    assert "mcp" in refreshed.errors
