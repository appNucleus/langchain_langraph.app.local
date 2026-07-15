from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from app.factory import create_app
from app.graph import ChatAgent
from app.services.inventory import ModelSelector
from app.settings import Settings


def _test_settings(*, api_key: str = "") -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="echo",
        mcp_enabled=False,
        api_key=api_key,
        ollama_required=False,
        mcp_required=False,
        persistence_required=False,
        artifact_storage_required=False,
        final_verification_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )


@contextmanager
def _client(*, api_key: str = "") -> Iterator[TestClient]:
    settings = _test_settings(api_key=api_key)
    app = create_app(settings=settings, chat_agent=ChatAgent(settings))
    with TestClient(app) as client:
        yield client


def test_health_endpoints_report_process_and_readiness() -> None:
    with _client() as client:
        health = client.get("/health")
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_chat_uses_current_worker_verifier_contract() -> None:
    with _client() as client:
        response = client.post(
            "/api/chat",
            json={"message": "hello", "system_prompt": ""},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "echo"
    assert payload["model"] == "echo"
    assert payload["response"]
    assert payload["metadata"]["plan"]["tasks"]
    assert payload["metadata"]["verification"]
    assert payload["metadata"]["termination_reason"] is None

    usage = payload["metadata"]["usage"]
    assert usage["model_calls"] == usage["physical_model_attempts"]
    assert usage["tool_calls"] == usage["tool_attempts"]
    assert usage["verifier_rounds"] >= 0
    assert usage["schema_version"] == 2
    assert "fallback_attempts" in usage
    assert "model_failures" in usage
    assert "tool_timeouts" in usage
    assert "deadline_at" in usage
    assert "elapsed_seconds" in usage


def test_chat_accepts_omitted_optional_system_prompt() -> None:
    with _client() as client:
        response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200
    prompt_metadata = response.json()["metadata"]["system_prompt"]
    assert prompt_metadata["generated"] is True
    assert prompt_metadata["source"] == "derived"


def test_api_key_is_enforced() -> None:
    with _client(api_key="secret") as client:
        missing = client.post("/api/chat", json={"message": "hello"})
        accepted = client.post(
            "/api/chat",
            json={"message": "hello"},
            headers={"X-API-Key": "secret"},
        )

    assert missing.status_code == 401
    assert accepted.status_code == 200


def test_chat_validation_rejects_empty_message() -> None:
    with _client() as client:
        response = client.post("/api/chat", json={"message": ""})

    assert response.status_code == 422


def test_stream_endpoint_returns_valid_sse() -> None:
    with _client() as client:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "hello"},
        ) as response:
            text = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: request_started" in text
    assert "event: completed" in text
    assert "data:" in text
    assert "Echo mode is active" in text


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


def _inventory_app() -> tuple[TestClient, FakeAgent]:
    settings = Settings(
        _env_file=None,
        llm_backend="ollama",
        mcp_enabled=True,
        inventory_cache_ttl_seconds=60,
        ollama_required=False,
        mcp_required=False,
        persistence_required=False,
        artifact_storage_required=False,
    )
    agent = FakeAgent(settings)
    app = create_app(settings=settings, chat_agent=agent)  # type: ignore[arg-type]
    return TestClient(app), agent


def test_inventory_endpoint_uses_shared_inventory_cache() -> None:
    client, agent = _inventory_app()
    with client:
        first = client.get("/api/inventory")
        second = client.get("/api/inventory")

    assert first.status_code == second.status_code == 200
    assert first.json()["cache"]["cached"] is False
    assert second.json()["cache"]["cached"] is True
    assert agent.ollama.calls == 1
    assert agent.mcp.calls == 1


def test_readiness_reuses_inventory_cache_and_liveness_is_process_only() -> None:
    client, agent = _inventory_app()
    with client:
        ready_one = client.get("/health/ready")
        ready_two = client.get("/health/ready")
        live = client.get("/health/live")

    assert ready_one.status_code == ready_two.status_code == 200
    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert agent.ollama.calls == 1
    assert agent.mcp.calls == 1
