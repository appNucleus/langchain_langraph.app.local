from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.orchestration.run_context import RunIdentity
from app.schemas.chat import ChatRequest, ChatResponse


def test_legacy_thread_id_maps_to_conversation_but_not_run() -> None:
    request = ChatRequest(message="hello", thread_id="conversation-1")
    identity = RunIdentity.resolve(
        conversation_id=request.conversation_id,
        run_id=request.run_id,
        legacy_thread_id=request.thread_id,
    )
    assert identity.conversation_id == "conversation-1"
    assert identity.run_id
    assert identity.execution_thread_id == f"conversation-1:{identity.run_id}"


def test_two_runs_have_distinct_execution_threads() -> None:
    first = RunIdentity.resolve(conversation_id="conversation-1")
    second = RunIdentity.resolve(conversation_id="conversation-1")
    assert first.conversation_id == second.conversation_id
    assert first.run_id != second.run_id
    assert first.execution_thread_id != second.execution_thread_id


def test_conflicting_legacy_and_canonical_identity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            message="hello",
            thread_id="one",
            conversation_id="two",
        )


def test_legacy_response_construction_remains_compatible() -> None:
    response = ChatResponse(
        thread_id="conversation-1",
        response="ok",
        backend="test",
    )
    assert response.conversation_id == "conversation-1"
    assert response.run_id
    assert response.execution_thread_id == f"conversation-1:{response.run_id}"
