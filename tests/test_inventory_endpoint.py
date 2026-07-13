from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.factory import create_app
from app.services.inventory import ModelSelector
from app.settings import Settings


class FakeOllama:
    def __init__(self) -> None:
        self.calls = 0

    async def list_models(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return [{"name": "qwen3.5:4b"}]

    async def health(self) -> dict[str, Any]:
        return {"status": "available"}


class FakeMCP:
    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return [{"name": "health_check", "inputSchema": {"type": "object"}}]

    async def health_check(self) -> Any:
        return type("Result", (), {"ok": True, "error": None})()


class FakeAgent:
    def __init__(self, settings: Settings) -> None:
        self.ollama = FakeOllama()
        self.mcp = FakeMCP()
        self.selector = ModelSelector(settings)

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def persistence_health(self) -> dict[str, Any]:
        return {"status": "available", "backend": "memory"}


def test_inventory_endpoint_uses_shared_inventory_service_cache() -> None:
    settings = Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        inventory_cache_ttl_seconds=60,
    )
    agent = FakeAgent(settings)
    app = create_app(settings=settings, chat_agent=agent)  # type: ignore[arg-type]

    with TestClient(app) as client:
        first = client.get("/api/inventory")
        second = client.get("/api/inventory")

    assert first.status_code == second.status_code == 200
    assert first.json()["cache"]["cached"] is False
    assert second.json()["cache"]["cached"] is True
    assert agent.ollama.calls == 1
    assert agent.mcp.calls == 1


def test_readiness_uses_cached_inventory_and_liveness_is_process_only() -> None:
    settings = Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        inventory_cache_ttl_seconds=60,
    )
    agent = FakeAgent(settings)
    app = create_app(settings=settings, chat_agent=agent)  # type: ignore[arg-type]

    with TestClient(app) as client:
        ready_one = client.get("/health/ready")
        ready_two = client.get("/health/ready")
        live = client.get("/health/live")

    assert ready_one.status_code == ready_two.status_code == 200
    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert agent.ollama.calls == 1
    assert agent.mcp.calls == 1
