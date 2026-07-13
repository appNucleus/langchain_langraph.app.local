from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import hmac
import json
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic, time
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.run import RunIdentity
from app.settings import Settings

_RESERVED_METADATA_KEYS = {
    "conversation_id",
    "execution_thread_id",
    "resume",
    "resume_token",
    "run_id",
    "thread_id",
}


class Phase5RequestError(RuntimeError):
    """Safe API-facing Phase 5 request error."""

    status_code = 400
    error_code = "phase5_request_error"


class ConversationBusyError(Phase5RequestError):
    status_code = 409
    error_code = "conversation_busy"


class RunConflictError(Phase5RequestError):
    status_code = 409
    error_code = "run_conflict"


class ResumeTokenInvalidError(Phase5RequestError):
    status_code = 401
    error_code = "invalid_resume_token"


class ResumeTokenExpiredError(Phase5RequestError):
    status_code = 410
    error_code = "resume_expired"


class ResumeRunNotFoundError(Phase5RequestError):
    status_code = 404
    error_code = "resumable_run_not_found"


class IncompatibleRunStateError(Phase5RequestError):
    status_code = 409
    error_code = "incompatible_run_state"


@dataclass(frozen=True)
class _RunRecord:
    request_hash: str
    status: str
    response: dict[str, Any] | None = None


class RunIdentityService:
    """Normalize legacy IDs and issue argument-bound signed resume tokens."""

    def __init__(self, settings: Settings) -> None:
        configured_secret = str(settings.resume_token_secret or "").strip()
        api_key_secret = str(settings.api_key or "").strip()
        secret = configured_secret or api_key_secret
        self._persistent_secret = bool(secret)
        self._secret = (secret or secrets.token_urlsafe(48)).encode("utf-8")
        self._ttl_seconds = int(settings.resume_token_ttl_seconds)
        self._checkpoint_namespace = settings.run_checkpoint_namespace
        self._state_schema_version = int(settings.run_state_schema_version)

    @property
    def token_ttl_seconds(self) -> int:
        return self._ttl_seconds

    @property
    def token_persistent(self) -> bool:
        """Whether tokens survive process restart with the configured secret."""

        return self._persistent_secret

    def sanitized_metadata(self, request: ChatRequest) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in request.metadata.items()
            if str(key).lower() not in _RESERVED_METADATA_KEYS
        }

    def request_hash(self, request: ChatRequest) -> str:
        canonical = json.dumps(
            {
                "message": request.message,
                "system_prompt": request.system_prompt,
                "metadata": self.sanitized_metadata(request),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def normalize(self, request: ChatRequest) -> RunIdentity:
        request_hash = self.request_hash(request)
        if request.resume:
            return self._identity_from_token(request, request_hash=request_hash)

        conversation_id = (
            request.conversation_id or request.thread_id or str(uuid4())
        )
        supplied_run_id = bool(request.run_id)
        run_id = request.run_id or str(uuid4())
        self._validate_uuid(run_id, field_name="run_id")
        execution_thread_id = f"{conversation_id}:{run_id}"
        return RunIdentity(
            conversation_id=conversation_id,
            run_id=run_id,
            execution_thread_id=execution_thread_id,
            checkpoint_namespace=self._checkpoint_namespace,
            state_schema_version=self._state_schema_version,
            request_hash=request_hash,
            resume_requested=False,
            resumed=False,
            client_supplied_run_id=supplied_run_id,
        )

    def issue_resume_token(self, identity: RunIdentity) -> str:
        now = int(time())
        payload = {
            "v": 1,
            "conversation_id": identity.conversation_id,
            "run_id": identity.run_id,
            "execution_thread_id": identity.execution_thread_id,
            "checkpoint_namespace": identity.checkpoint_namespace,
            "state_schema_version": identity.state_schema_version,
            "request_hash": identity.request_hash,
            "iat": now,
            "exp": now + self._ttl_seconds,
        }
        encoded_payload = self._b64encode(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        signature = self._b64encode(
            hmac.new(
                self._secret,
                encoded_payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
        )
        return f"{encoded_payload}.{signature}"

    def _identity_from_token(
        self,
        request: ChatRequest,
        *,
        request_hash: str,
    ) -> RunIdentity:
        token = str(request.resume_token or "")
        payload = self._decode_token(token)
        if int(payload.get("v", 0)) != 1:
            raise ResumeTokenInvalidError("Unsupported resume token version")

        expires_at = int(payload.get("exp", 0) or 0)
        if expires_at <= int(time()):
            raise ResumeTokenExpiredError("The resume token has expired")

        token_schema_version = int(payload.get("state_schema_version", 0) or 0)
        if token_schema_version != self._state_schema_version:
            raise IncompatibleRunStateError(
                "The resumable run uses an incompatible state schema version"
            )

        conversation_id = str(payload.get("conversation_id") or "").strip()
        run_id = str(payload.get("run_id") or "").strip()
        execution_thread_id = str(payload.get("execution_thread_id") or "").strip()
        checkpoint_namespace = str(payload.get("checkpoint_namespace") or "").strip()
        token_request_hash = str(payload.get("request_hash") or "").strip()
        if not all(
            (
                conversation_id,
                run_id,
                execution_thread_id,
                checkpoint_namespace,
                token_request_hash,
            )
        ):
            raise ResumeTokenInvalidError("Resume token is missing required identity data")

        self._validate_uuid(run_id, field_name="run_id")
        if execution_thread_id != f"{conversation_id}:{run_id}":
            raise ResumeTokenInvalidError("Resume token execution identity is invalid")
        if checkpoint_namespace != self._checkpoint_namespace:
            raise IncompatibleRunStateError(
                "The resumable run uses an incompatible checkpoint namespace"
            )
        if token_request_hash != request_hash:
            raise ResumeTokenInvalidError(
                "The resume token is not valid for this request payload"
            )

        supplied_conversation_id = request.conversation_id or request.thread_id
        if supplied_conversation_id and supplied_conversation_id != conversation_id:
            raise ResumeTokenInvalidError(
                "The resume token does not match the supplied conversation ID"
            )
        if request.run_id and request.run_id != run_id:
            raise ResumeTokenInvalidError(
                "The resume token does not match the supplied run ID"
            )

        return RunIdentity(
            conversation_id=conversation_id,
            run_id=run_id,
            execution_thread_id=execution_thread_id,
            checkpoint_namespace=checkpoint_namespace,
            state_schema_version=token_schema_version,
            request_hash=request_hash,
            resume_requested=True,
            resumed=True,
            client_supplied_run_id=bool(request.run_id),
        )

    def _decode_token(self, token: str) -> dict[str, Any]:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
        except ValueError as exc:
            raise ResumeTokenInvalidError("Malformed resume token") from exc

        expected_signature = hmac.new(
            self._secret,
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        try:
            supplied_signature = self._b64decode(encoded_signature)
        except (ValueError, UnicodeError) as exc:
            raise ResumeTokenInvalidError("Malformed resume token signature") from exc
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ResumeTokenInvalidError("Resume token signature is invalid")

        try:
            payload = json.loads(self._b64decode(encoded_payload).decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResumeTokenInvalidError("Malformed resume token payload") from exc
        if not isinstance(payload, dict):
            raise ResumeTokenInvalidError("Resume token payload must be an object")
        return payload

    @staticmethod
    def _validate_uuid(value: str, *, field_name: str) -> None:
        try:
            UUID(value)
        except ValueError as exc:
            raise ResumeTokenInvalidError(f"{field_name} must be a valid UUID") from exc

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))


class LocalConversationGate:
    """Reject overlapping turns for one conversation in this app process."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, identity: RunIdentity) -> AsyncIterator[None]:
        async with self._lock:
            owner = self._owners.get(identity.conversation_id)
            if owner is not None:
                raise ConversationBusyError(
                    "Another run is already active for this conversation"
                )
            self._owners[identity.conversation_id] = identity.run_id
        try:
            yield
        finally:
            async with self._lock:
                if self._owners.get(identity.conversation_id) == identity.run_id:
                    self._owners.pop(identity.conversation_id, None)


class LocalRunRegistry:
    """Bounded process-local idempotency cache pending the Phase 8 repository."""

    def __init__(self, *, ttl_seconds: int, max_records: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_records = max_records
        self._records: OrderedDict[
            tuple[str, str], tuple[float, _RunRecord]
        ] = OrderedDict()
        self._lock = asyncio.Lock()

    async def cached_response(self, identity: RunIdentity) -> ChatResponse | None:
        key = (identity.conversation_id, identity.run_id)
        async with self._lock:
            self._purge(monotonic())
            stored = self._records.get(key)
            if stored is None:
                return None
            _, record = stored
            self._assert_same_request(record, identity)
            self._touch(key, record)
            if record.status == "completed" and record.response is not None:
                return ChatResponse.model_validate(record.response)
            return None

    async def start(self, identity: RunIdentity) -> None:
        key = (identity.conversation_id, identity.run_id)
        async with self._lock:
            self._purge(monotonic())
            stored = self._records.get(key)
            if stored is not None:
                _, record = stored
                self._assert_same_request(record, identity)
                if record.status == "completed":
                    raise RunConflictError("The run is already completed")
                if record.status == "running":
                    raise ConversationBusyError("The run is already active")
                if not identity.resumed:
                    raise RunConflictError(
                        "The run already exists; use its resume token to continue it"
                    )
            self._store(
                key,
                _RunRecord(
                    request_hash=identity.request_hash,
                    status="running",
                ),
            )

    async def complete(self, identity: RunIdentity, response: ChatResponse) -> None:
        key = (identity.conversation_id, identity.run_id)
        async with self._lock:
            self._purge(monotonic())
            self._store(
                key,
                _RunRecord(
                    request_hash=identity.request_hash,
                    status="completed",
                    response=response.model_dump(mode="json"),
                ),
            )

    async def interrupt(self, identity: RunIdentity) -> None:
        key = (identity.conversation_id, identity.run_id)
        async with self._lock:
            self._purge(monotonic())
            stored = self._records.get(key)
            if stored is None:
                return
            _, record = stored
            if record.status == "completed":
                return
            self._store(
                key,
                _RunRecord(
                    request_hash=identity.request_hash,
                    status="interrupted",
                ),
            )

    def _store(self, key: tuple[str, str], record: _RunRecord) -> None:
        self._records[key] = (monotonic(), record)
        self._records.move_to_end(key)
        while len(self._records) > self._max_records:
            removable = next(
                (
                    candidate
                    for candidate, (_, item) in self._records.items()
                    if item.status != "running"
                ),
                None,
            )
            if removable is None:
                break
            self._records.pop(removable, None)

    def _touch(self, key: tuple[str, str], record: _RunRecord) -> None:
        self._records[key] = (monotonic(), record)
        self._records.move_to_end(key)

    def _purge(self, now: float) -> None:
        cutoff = now - self._ttl_seconds
        expired = [
            key
            for key, (timestamp, record) in self._records.items()
            if timestamp < cutoff and record.status != "running"
        ]
        for key in expired:
            self._records.pop(key, None)

    @staticmethod
    def _assert_same_request(record: _RunRecord, identity: RunIdentity) -> None:
        if record.request_hash != identity.request_hash:
            raise RunConflictError(
                "The supplied run_id is already bound to a different request payload"
            )

