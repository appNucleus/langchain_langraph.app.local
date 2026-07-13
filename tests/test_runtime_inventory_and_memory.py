from __future__ import annotations

from typing import Any

import pytest

from app.services.inventory import InventoryService
from app.settings import Settings
from app.state.in_memory import BoundedInMemoryStore


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
async def test_inventory_cache_is_single_flight_and_returns_defensive_copies() -> None:
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
async def test_inventory_uses_stale_failed_source_without_discarding_fresh_source() -> None:
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


@pytest.mark.asyncio
async def test_bounded_memory_limits_messages_and_evicts_lru_sessions() -> None:
    store = BoundedInMemoryStore(
        ttl_seconds=60,
        max_sessions=2,
        max_messages=2,
    )
    await store.append("a", {"content": "1"}, {"content": "2"}, {"content": "3"})
    await store.append("b", {"content": "b"})
    assert [item["content"] for item in await store.get("a")] == ["2", "3"]

    # Reading a makes b the least-recently-used session.
    await store.append("c", {"content": "c"})
    assert await store.get("b") == []
    assert (await store.get("a"))[0]["content"] == "2"
    assert (await store.get("c"))[0]["content"] == "c"


@pytest.mark.asyncio
async def test_bounded_memory_ttl_uses_inactivity(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("app.state.in_memory.monotonic", lambda: now)
    store = BoundedInMemoryStore(
        ttl_seconds=10,
        max_sessions=2,
        max_messages=2,
    )
    await store.append("thread", {"content": "x"})

    now = 109.0
    assert await store.get("thread")
    now = 118.0
    assert await store.get("thread")  # read at 109 refreshed inactivity TTL
    now = 129.0
    assert await store.get("thread") == []
