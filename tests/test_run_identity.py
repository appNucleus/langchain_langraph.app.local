from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from app.orchestration.run_identity import (
    ConversationBusyError,
    ConversationGate,
    ResumeTokenInvalidError,
    RunIdentityService,
)
from app.schemas.chat import ChatRequest
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="echo",
        ollama_required=False,
        mcp_required=False,
        resume_token_secret="test-secret",
    )


def test_message_only_request_generates_new_conversation_and_run_ids() -> None:
    service = RunIdentityService(_settings())
    first = service.normalize(ChatRequest(message="Continue the analysis"))
    second = service.normalize(ChatRequest(message="Continue the analysis"))
    assert first.conversation_id != second.conversation_id
    assert first.run_id != second.run_id
    assert first.execution_thread_id == f"{first.conversation_id}:{first.run_id}"


def test_langgraph_config_is_json_safe() -> None:
    identity = RunIdentityService(_settings()).normalize(
        ChatRequest(message="hello", conversation_id="conversation-postgres")
    )
    config = identity.langgraph_config()
    assert config["configurable"] == {"thread_id": identity.execution_thread_id}
    assert config["metadata"]["run_id"] == identity.run_id
    assert json.loads(json.dumps(config)) == config


def test_legacy_thread_id_is_normalized_as_conversation_id() -> None:
    identity = RunIdentityService(_settings()).normalize(
        ChatRequest(message="hello", thread_id="conversation-123")
    )
    assert identity.conversation_id == "conversation-123"


def test_explicit_values_replace_defaults() -> None:
    service = RunIdentityService(_settings())
    run_id = str(uuid4())
    request = ChatRequest(
        message="hello",
        conversation_id="conversation-explicit",
        run_id=run_id,
        metadata={"client": "postman", "run_id": "must-not-win"},
    )
    identity = service.normalize(request)
    assert identity.run_id == run_id
    assert service.sanitized_metadata(request) == {"client": "postman"}


def test_thread_and_conversation_must_match() -> None:
    with pytest.raises(ValueError, match="must match"):
        ChatRequest(
            message="hello",
            thread_id="legacy-a",
            conversation_id="conversation-b",
        )


def test_resume_token_is_bound_to_request_and_identity() -> None:
    service = RunIdentityService(_settings())
    original = ChatRequest(
        message="Continue the analysis",
        conversation_id="conversation-123",
        metadata={"scope": "identity"},
    )
    identity = service.normalize(original)
    token = service.issue_resume_token(identity)
    resumed = service.normalize(
        ChatRequest(
            message="Continue the analysis",
            conversation_id="conversation-123",
            run_id=identity.run_id,
            resume=True,
            resume_token=token,
            metadata={"scope": "identity"},
        )
    )
    assert resumed.resumed is True
    assert resumed.run_id == identity.run_id


def test_resume_token_rejects_changed_request_payload() -> None:
    service = RunIdentityService(_settings())
    identity = service.normalize(ChatRequest(message="original"))
    token = service.issue_resume_token(identity)
    with pytest.raises(ResumeTokenInvalidError, match="not valid"):
        service.normalize(
            ChatRequest(
                message="changed",
                resume=True,
                resume_token=token,
            )
        )


@pytest.mark.asyncio
async def test_same_conversation_gate_rejects_overlapping_run() -> None:
    service = RunIdentityService(_settings())
    gate = ConversationGate()
    first = service.normalize(
        ChatRequest(message="first", conversation_id="shared-conversation")
    )
    second = service.normalize(
        ChatRequest(message="second", conversation_id="shared-conversation")
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_first() -> None:
        async with gate.hold(first):
            entered.set()
            await release.wait()

    task = asyncio.create_task(hold_first())
    await entered.wait()
    try:
        with pytest.raises(ConversationBusyError):
            async with gate.hold(second):
                pass
    finally:
        release.set()
        await task
