from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from app.orchestration.run_identity import (
    ConversationBusyError,
    RunIdentityService,
    StaleLeaseError,
)
from app.schemas.chat import ChatRequest
from app.settings import Settings
from app.state.postgres import PostgresConversationStore
from app.state.run_repository import PostgresRunRepository

pytestmark = pytest.mark.integration


def _database_url() -> str:
    value = os.getenv("TEST_DATABASE_URL", "").strip()
    if not value:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    return value


def _identity_service() -> RunIdentityService:
    return RunIdentityService(
        Settings(
            _env_file=None,
            llm_backend="echo",
            ollama_required=False,
            mcp_required=False,
            resume_token_secret="integration-secret",
        )
    )


def _repository(database_url: str) -> PostgresRunRepository:
    return PostgresRunRepository(
        database_url,
        min_pool_size=1,
        max_pool_size=2,
        command_timeout=10,
        auto_setup=True,
    )


@pytest.mark.asyncio
async def test_postgres_lease_fencing_and_restart_safe_replay() -> None:
    database_url = _database_url()
    first_repository = _repository(database_url)
    second_repository = _repository(database_url)
    await first_repository.start()
    await second_repository.start()
    service = _identity_service()
    conversation_id = f"integration-{uuid4()}"
    first = service.normalize(
        ChatRequest(message="first", conversation_id=conversation_id)
    )
    second = service.normalize(
        ChatRequest(message="second", conversation_id=conversation_id)
    )
    try:
        await first_repository.create_or_get(first)
        await second_repository.create_or_get(second)
        stale = await first_repository.acquire(
            first,
            owner_id=str(uuid4()),
            ttl_seconds=1,
        )
        with pytest.raises(ConversationBusyError):
            await second_repository.acquire(
                second,
                owner_id=str(uuid4()),
                ttl_seconds=30,
            )

        await asyncio.sleep(1.1)
        replacement = await second_repository.acquire(
            second,
            owner_id=str(uuid4()),
            ttl_seconds=30,
        )
        assert replacement.fencing_token > stale.fencing_token
        with pytest.raises(StaleLeaseError):
            await first_repository.mark_terminal(
                stale,
                status="completed",
                response_payload={"response": "stale"},
                termination_reason=None,
                error_code=None,
            )

        payload = {
            "thread_id": second.conversation_id,
            "conversation_id": second.conversation_id,
            "run_id": second.run_id,
            "response": "durable",
            "backend": "echo",
            "model": None,
            "metadata": {},
        }
        await second_repository.mark_terminal(
            replacement,
            status="completed",
            response_payload=payload,
            termination_reason=None,
            error_code=None,
        )
        await second_repository.release(replacement)
    finally:
        await first_repository.aclose()
        await second_repository.aclose()

    restarted = _repository(database_url)
    await restarted.start()
    try:
        stored = await restarted.get(second.run_id)
        assert stored is not None
        assert stored.status == "completed"
        assert stored.response_payload == payload
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_postgres_history_append_is_idempotent_by_run_id() -> None:
    database_url = _database_url()
    store = PostgresConversationStore(
        database_url,
        min_pool_size=1,
        max_pool_size=2,
        max_messages=10,
        command_timeout=10,
    )
    await store.start()
    conversation_id = f"history-{uuid4()}"
    run_id = str(uuid4())
    user = {"role": "user", "content": "hello", "metadata": {"run_id": run_id}}
    assistant = {
        "role": "assistant",
        "content": "hi",
        "metadata": {"run_id": run_id},
    }
    try:
        assert await store.append_turn(
            conversation_id,
            run_id=run_id,
            user_message=user,
            assistant_message=assistant,
        )
        assert not await store.append_turn(
            conversation_id,
            run_id=run_id,
            user_message=user,
            assistant_message=assistant,
        )
        history = await store.get(conversation_id)
        assert [item["role"] for item in history] == ["user", "assistant"]
    finally:
        await store.clear(conversation_id)
        await store.aclose()


@pytest.mark.asyncio
async def test_postgres_history_idempotency_survives_visible_trimming() -> None:
    database_url = _database_url()
    store = PostgresConversationStore(
        database_url,
        min_pool_size=1,
        max_pool_size=2,
        max_messages=2,
        command_timeout=10,
    )
    await store.start()
    conversation_id = f"history-trim-{uuid4()}"
    first_run = str(uuid4())
    second_run = str(uuid4())

    def turn(run_id: str, content: str) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {"role": "user", "content": content, "metadata": {"run_id": run_id}},
            {
                "role": "assistant",
                "content": f"answer-{content}",
                "metadata": {"run_id": run_id},
            },
        )

    first_user, first_assistant = turn(first_run, "first")
    second_user, second_assistant = turn(second_run, "second")
    try:
        assert await store.append_turn(
            conversation_id,
            run_id=first_run,
            user_message=first_user,
            assistant_message=first_assistant,
        )
        assert await store.append_turn(
            conversation_id,
            run_id=second_run,
            user_message=second_user,
            assistant_message=second_assistant,
        )
        assert not await store.append_turn(
            conversation_id,
            run_id=first_run,
            user_message=first_user,
            assistant_message=first_assistant,
        )
        history = await store.get(conversation_id)
        assert [item["content"] for item in history] == [
            "second",
            "answer-second",
        ]
    finally:
        await store.clear(conversation_id)
        await store.aclose()

