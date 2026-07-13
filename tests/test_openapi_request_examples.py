from __future__ import annotations

import json
from typing import Any

from app.factory import create_app
from app.schemas.chat import (
    ChatRequest,
    build_chat_request_openapi_examples,
    load_chat_request_example,
)
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
        run_repository_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
        resume_token_secret="test-secret",
    )


def test_chat_request_defaults_are_defined_by_the_model() -> None:
    request = ChatRequest(message="Continue the analysis")
    assert request.model_dump(mode="json") == {
        "message": "Continue the analysis",
        "thread_id": None,
        "conversation_id": None,
        "run_id": None,
        "resume": False,
        "resume_token": None,
        "system_prompt": None,
        "metadata": {},
    }


def test_documentation_loader_validates_code_supplied_content(
    monkeypatch: Any,
) -> None:
    source = {
        "message": "Documentation-only request example",
        "metadata": {"source": "code-defined-test-data"},
    }
    monkeypatch.setattr(
        "app.schemas.chat.Path.read_text",
        lambda _path, *, encoding: json.dumps(source),
    )

    assert load_chat_request_example() == source


def test_create_app_loads_documentation_example_only_for_openapi(
    monkeypatch: Any,
) -> None:
    calls = 0

    def load_documentation_example() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"message": "Documentation-only request example"}

    monkeypatch.setattr(
        "app.factory.load_chat_request_example",
        load_documentation_example,
    )

    app = create_app(settings=_settings())
    assert calls == 0

    app.openapi()
    assert calls == 1

    app.openapi()
    assert calls == 1


def test_openapi_uses_code_injected_documentation_example(
    in_memory_chat_request_example: dict[str, Any],
) -> None:
    schema = create_app(settings=_settings()).openapi()
    for path in ("/api/chat", "/api/chat/stream"):
        examples = schema["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]
        assert examples["default"]["value"] == in_memory_chat_request_example


def test_openapi_example_builder_returns_a_defensive_copy() -> None:
    source = {"message": "example", "metadata": {"source": "in-memory"}}
    examples = build_chat_request_openapi_examples(source)
    examples["default"]["value"]["metadata"]["source"] = "mutated"
    assert source["metadata"]["source"] == "in-memory"
