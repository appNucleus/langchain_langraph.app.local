from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.factory import create_app
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="echo",
        mcp_enabled=False,
        ollama_required=False,
        mcp_required=False,
        persistence_required=False,
        artifact_storage_required=False,
        final_verification_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
        resume_token_secret="test-secret",
    )


def test_message_only_request_starts_new_conversation() -> None:
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Continue the analysis"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"]
    assert payload["conversation_id"] == payload["thread_id"]
    assert payload["run_id"]
    assert payload["metadata"]["phase"] == "5"
    identity = payload["metadata"]["identity"]
    assert identity["execution_thread_id"] == (
        f"{payload['conversation_id']}:{payload['run_id']}"
    )
    assert identity["resume_token"]


def test_sequential_turns_share_conversation_but_not_checkpoint_thread() -> None:
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        first = client.post("/api/chat", json={"message": "first"}).json()
        second = client.post(
            "/api/chat",
            json={"message": "second", "thread_id": first["thread_id"]},
        ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert second["run_id"] != first["run_id"]
    assert (
        second["metadata"]["identity"]["execution_thread_id"]
        != first["metadata"]["identity"]["execution_thread_id"]
    )


def test_explicit_run_id_is_idempotent_within_process() -> None:
    app = create_app(settings=_settings())
    run_id = str(uuid4())
    body = {
        "message": "same request",
        "conversation_id": "conversation-idempotent",
        "run_id": run_id,
    }
    with TestClient(app) as client:
        first = client.post("/api/chat", json=body)
        second = client.post("/api/chat", json=body)
        history = client.app.state.chat_agent.memory
        stored = client.portal.call(history.get, "conversation-idempotent")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(stored) == 2


def test_reusing_run_id_for_different_payload_returns_conflict() -> None:
    app = create_app(settings=_settings())
    run_id = str(uuid4())
    with TestClient(app) as client:
        first = client.post(
            "/api/chat",
            json={
                "message": "first payload",
                "conversation_id": "conversation-conflict",
                "run_id": run_id,
            },
        )
        conflict = client.post(
            "/api/chat",
            json={
                "message": "different payload",
                "conversation_id": "conversation-conflict",
                "run_id": run_id,
            },
        )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "run_conflict"


def test_openapi_uses_stored_complete_request_example() -> None:
    app = create_app(settings=_settings())
    schema = app.openapi()
    examples = schema["paths"]["/api/chat"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]

    assert examples["default"]["value"] == {
        "message": "Continue the analysis",
        "thread_id": None,
        "conversation_id": None,
        "run_id": None,
        "resume": False,
        "resume_token": None,
        "system_prompt": None,
        "metadata": {},
    }


def test_stream_starts_with_resumable_identity() -> None:
    app = create_app(settings=_settings())
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "Continue the analysis"},
        ) as response:
            text = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: request_started" in text
    assert '"conversation_id"' in text
    assert '"run_id"' in text
    assert '"resume_token"' in text
