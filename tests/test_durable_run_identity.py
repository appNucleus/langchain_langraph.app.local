from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from uuid import uuid4

import pytest

from app.orchestration.run_identity import (
    ConversationBusyError,
    RunIdentityService,
    StaleLeaseError,
)
from app.schemas.chat import ChatRequest
from app.settings import Settings
from app.state.run_repository import MemoryRunRepository


def _settings(**overrides: object) -> Settings:
    values = {
        "_env_file": None,
        "llm_backend": "echo",
        "ollama_required": False,
        "mcp_required": False,
        "resume_token_secret": "test-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_request_hash_is_canonical_and_reserved_metadata_is_ignored() -> None:
    service = RunIdentityService(_settings())
    first = ChatRequest(
        message="café",
        metadata={"b": 2, "a": [1, {"z": True}], "run_id": "caller-value"},
    )
    second = ChatRequest(
        message="cafe\u0301",
        metadata={"a": [1, {"z": True}], "b": 2},
    )
    assert service.request_hash(first) == service.request_hash(second)


def test_resume_token_key_rotation_accepts_previous_key() -> None:
    old = RunIdentityService(
        _settings(
            resume_token_active_key_id="old",
            resume_token_keys_json=json.dumps({"old": "old-secret"}),
        )
    )
    request = ChatRequest(message="continue", conversation_id="conversation-1")
    identity = old.normalize(request)
    token = old.issue_resume_token(identity)

    rotated = RunIdentityService(
        _settings(
            resume_token_active_key_id="new",
            resume_token_keys_json=json.dumps(
                {"new": "new-secret", "old": "old-secret"}
            ),
        )
    )
    resumed = rotated.normalize(
        ChatRequest(
            message="continue",
            conversation_id="conversation-1",
            run_id=identity.run_id,
            resume=True,
            resume_token=token,
        )
    )
    assert resumed.run_id == identity.run_id
    assert resumed.resume_token_key_id == "old"


def test_legacy_v1_resume_token_remains_request_bound() -> None:
    service = RunIdentityService(_settings())
    request = ChatRequest(
        message="continue",
        conversation_id="legacy-conversation",
        metadata={"scope": "legacy"},
    )
    identity = service.normalize(request)
    legacy_canonical = json.dumps(
        {
            "message": request.message,
            "system_prompt": request.system_prompt,
            "metadata": service.sanitized_metadata(request),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    now = int(time.time())
    payload = {
        "v": 1,
        "conversation_id": identity.conversation_id,
        "run_id": identity.run_id,
        "execution_thread_id": identity.execution_thread_id,
        "checkpoint_namespace": identity.checkpoint_namespace,
        "state_schema_version": identity.state_schema_version,
        "request_hash": hashlib.sha256(legacy_canonical.encode()).hexdigest(),
        "iat": now,
        "exp": now + 60,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).rstrip(b"=")
    signature = base64.urlsafe_b64encode(
        hmac.new(b"test-secret", encoded, hashlib.sha256).digest()
    ).rstrip(b"=")
    token = f"{encoded.decode()}.{signature.decode()}"

    resumed = service.normalize(
        ChatRequest(
            message="continue",
            conversation_id="legacy-conversation",
            run_id=identity.run_id,
            resume=True,
            resume_token=token,
            metadata={"scope": "legacy"},
        )
    )
    assert resumed.resumed is True
    assert resumed.request_hash == identity.request_hash
    assert resumed.request_hash_version == identity.request_hash_version


@pytest.mark.asyncio
async def test_memory_repository_replays_completed_response() -> None:
    service = RunIdentityService(_settings())
    identity = service.normalize(
        ChatRequest(
            message="same request",
            conversation_id="conversation-replay",
            run_id=str(uuid4()),
        )
    )
    repository = MemoryRunRepository()
    await repository.start()
    record, created = await repository.create_or_get(identity)
    assert created and record.status == "pending"
    lease = await repository.acquire(identity, owner_id=str(uuid4()), ttl_seconds=30)
    payload = {
        "thread_id": identity.conversation_id,
        "conversation_id": identity.conversation_id,
        "run_id": identity.run_id,
        "response": "done",
        "backend": "echo",
        "model": None,
        "metadata": {},
    }
    await repository.mark_terminal(
        lease,
        status="completed",
        response_payload=payload,
        termination_reason=None,
        error_code=None,
    )
    stored = await repository.get(identity.run_id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.response_payload == payload


@pytest.mark.asyncio
async def test_same_conversation_has_one_active_lease() -> None:
    service = RunIdentityService(_settings())
    first = service.normalize(
        ChatRequest(message="first", conversation_id="shared-conversation")
    )
    second = service.normalize(
        ChatRequest(message="second", conversation_id="shared-conversation")
    )
    repository = MemoryRunRepository()
    await repository.start()
    await repository.create_or_get(first)
    await repository.create_or_get(second)
    await repository.acquire(first, owner_id=str(uuid4()), ttl_seconds=30)
    with pytest.raises(ConversationBusyError):
        await repository.acquire(second, owner_id=str(uuid4()), ttl_seconds=30)


@pytest.mark.asyncio
async def test_expired_lease_takeover_fences_stale_owner() -> None:
    service = RunIdentityService(_settings())
    first = service.normalize(
        ChatRequest(message="first", conversation_id="takeover-conversation")
    )
    second = service.normalize(
        ChatRequest(message="second", conversation_id="takeover-conversation")
    )
    repository = MemoryRunRepository()
    await repository.start()
    await repository.create_or_get(first)
    await repository.create_or_get(second)
    stale = await repository.acquire(first, owner_id=str(uuid4()), ttl_seconds=1)
    await asyncio.sleep(1.05)
    replacement = await repository.acquire(second, owner_id=str(uuid4()), ttl_seconds=30)
    assert replacement.fencing_token > stale.fencing_token
    with pytest.raises(StaleLeaseError):
        await repository.mark_terminal(
            stale,
            status="completed",
            response_payload={},
            termination_reason=None,
            error_code=None,
        )


@pytest.mark.asyncio
async def test_resume_token_version_can_be_revoked_durably() -> None:
    service = RunIdentityService(_settings())
    identity = service.normalize(ChatRequest(message="continue"))
    repository = MemoryRunRepository()
    await repository.start()
    await repository.create_or_get(identity)
    assert await repository.revoke_resume_tokens(identity.run_id) == 2
    record = await repository.get(identity.run_id)
    assert record is not None and record.resume_token_version == 2
