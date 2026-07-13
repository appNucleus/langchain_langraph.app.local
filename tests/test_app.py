from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.factory import create_app
from app.graph import ChatAgent
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
    assert payload["metadata"]["phase"] == "4"
    assert payload["metadata"]["plan"]["tasks"]
    assert payload["metadata"]["verification"]
    assert payload["metadata"]["termination_reason"] is None


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
