from __future__ import annotations

from typing import Any

from app.factory import create_app
from app.schemas.chat import (
    CHAT_REQUEST_EXAMPLE_PATH,
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


def test_runtime_default_request_example_is_minimal_and_valid() -> None:
    example = load_chat_request_example(CHAT_REQUEST_EXAMPLE_PATH)
    assert example == {"message": "Continue the analysis"}
    assert ChatRequest.model_validate(example).message == "Continue the analysis"


def test_complete_request_example_is_separate_full_and_valid() -> None:
    complete_path = CHAT_REQUEST_EXAMPLE_PATH.with_name("chat-complete." + "json")
    complete = load_chat_request_example(complete_path)

    assert set(complete) == {
        "message",
        "thread_id",
        "conversation_id",
        "run_id",
        "resume",
        "resume_token",
        "system_prompt",
        "metadata",
    }
    assert complete != load_chat_request_example(CHAT_REQUEST_EXAMPLE_PATH)
    assert complete["system_prompt"]
    assert complete["metadata"]
    assert ChatRequest.model_validate(complete).message == complete["message"]


def test_openapi_uses_the_runtime_request_example(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "app.factory.load_chat_request_example",
        lambda: load_chat_request_example(CHAT_REQUEST_EXAMPLE_PATH),
    )
    schema = create_app(settings=_settings()).openapi()
    for path in ("/api/chat", "/api/chat/stream"):
        examples = schema["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]
        assert examples["default"]["value"] == {
            "message": "Continue the analysis"
        }


def test_openapi_example_builder_returns_a_defensive_copy() -> None:
    source = {"message": "example", "metadata": {"source": "in-memory"}}
    examples = build_chat_request_openapi_examples(source)
    examples["default"]["value"]["metadata"]["source"] = "mutated"
    assert source["metadata"]["source"] == "in-memory"
