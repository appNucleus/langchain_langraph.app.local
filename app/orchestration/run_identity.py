from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import secrets
import unicodedata
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic, time
from typing import Any
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
_REQUEST_HASH_DOMAIN = b"langchain-langraph-chat-request-v1\0"


class RequestIdentityError(RuntimeError):
    """Safe API-facing request identity error."""

    status_code = 400
    error_code = "request_identity_error"


class ConversationBusyError(RequestIdentityError):
    status_code = 409
    error_code = "conversation_busy"


class RunConflictError(RequestIdentityError):
    status_code = 409
    error_code = "run_conflict"


class StaleLeaseError(RequestIdentityError):
    status_code = 409
    error_code = "stale_lease"


class RunNotResumableError(RequestIdentityError):
    status_code = 409
    error_code = "run_not_resumable"


class ResumeTokenInvalidError(RequestIdentityError):
    status_code = 401
    error_code = "invalid_resume_token"


class ResumeTokenRevokedError(RequestIdentityError):
    status_code = 401
    error_code = "resume_revoked"


class ResumeTokenExpiredError(RequestIdentityError):
    status_code = 410
    error_code = "resume_expired"


class ResumeRunNotFoundError(RequestIdentityError):
    status_code = 404
    error_code = "resumable_run_not_found"


class IncompatibleRunStateError(RequestIdentityError):
    status_code = 409
    error_code = "incompatible_run_state"


@dataclass(frozen=True)
class _RunRecord:
    request_hash: str
    status: str
    response: dict[str, Any] | None = None


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("request metadata must not contain NaN or infinity")
        return value
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        return {
            _normalize_text(str(key)): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(
        f"request metadata contains unsupported value type: {type(value).__name__}"
    )


class RunIdentityService:
    """Normalize request IDs and issue request-bound, rotatable resume tokens."""

    def __init__(self, settings: Settings) -> None:
        self._ttl_seconds = int(settings.resume_token_ttl_seconds)
        self._checkpoint_namespace = settings.run_checkpoint_namespace
        self._state_schema_version = int(settings.run_state_schema_version)
        self._request_hash_version = int(settings.run_request_hash_version)
        self._active_key_id = str(settings.resume_token_active_key_id).strip() or "primary"
        self._keys, self._persistent_secret = self._load_keys(settings)
        if self._active_key_id not in self._keys:
            raise ValueError(
                "RESUME_TOKEN_ACTIVE_KEY_ID must identify a configured signing key"
            )

    @property
    def token_ttl_seconds(self) -> int:
        return self._ttl_seconds

    @property
    def token_persistent(self) -> bool:
        return self._persistent_secret

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def sanitized_metadata(self, request: ChatRequest) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in request.metadata.items()
            if str(key).lower() not in _RESERVED_METADATA_KEYS
        }

    def request_hash(self, request: ChatRequest) -> str:
        canonical = json.dumps(
            {
                "version": self._request_hash_version,
                "message": _normalize_text(request.message),
                "system_prompt": (
                    _normalize_text(request.system_prompt)
                    if request.system_prompt is not None
                    else None
                ),
                "metadata": _canonical_json_value(self.sanitized_metadata(request)),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(_REQUEST_HASH_DOMAIN + canonical).hexdigest()

    def _legacy_request_hash(self, request: ChatRequest) -> str:
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

        conversation_id = request.conversation_id or request.thread_id or str(uuid4())
        supplied_run_id = bool(request.run_id)
        run_id = request.run_id or str(uuid4())
        self._validate_uuid(run_id, field_name="run_id")
        return RunIdentity(
            conversation_id=conversation_id,
            run_id=run_id,
            execution_thread_id=f"{conversation_id}:{run_id}",
            checkpoint_namespace=self._checkpoint_namespace,
            state_schema_version=self._state_schema_version,
            request_hash=request_hash,
            request_hash_version=self._request_hash_version,
            resume_token_version=1,
            resume_token_key_id=self._active_key_id,
            resume_requested=False,
            resumed=False,
            client_supplied_run_id=supplied_run_id,
        )

    def issue_resume_token(self, identity: RunIdentity) -> str:
        now = int(time())
        payload = {
            "v": 2,
            "kid": self._active_key_id,
            "conversation_id": identity.conversation_id,
            "run_id": identity.run_id,
            "execution_thread_id": identity.execution_thread_id,
            "checkpoint_namespace": identity.checkpoint_namespace,
            "state_schema_version": identity.state_schema_version,
            "request_hash": identity.request_hash,
            "request_hash_version": identity.request_hash_version,
            "resume_token_version": identity.resume_token_version,
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
                self._keys[self._active_key_id],
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
        payload, key_id = self._decode_token(str(request.resume_token or ""))
        token_format_version = int(payload.get("v", 0))
        if token_format_version not in {1, 2}:
            raise ResumeTokenInvalidError("Unsupported resume token version")
        if int(payload.get("exp", 0) or 0) <= int(time()):
            raise ResumeTokenExpiredError("The resume token has expired")

        schema_version = int(payload.get("state_schema_version", 0) or 0)
        if schema_version != self._state_schema_version:
            raise IncompatibleRunStateError(
                "The resumable run uses an incompatible state schema version"
            )

        hash_version = int(
            payload.get("request_hash_version", self._request_hash_version)
            or self._request_hash_version
        )
        if token_format_version == 2 and hash_version != self._request_hash_version:
            raise IncompatibleRunStateError(
                "The resumable run uses an incompatible request hash version"
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
        expected_request_hash = (
            self._legacy_request_hash(request)
            if token_format_version == 1
            else request_hash
        )
        if token_request_hash != expected_request_hash:
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
            state_schema_version=schema_version,
            request_hash=request_hash,
            request_hash_version=self._request_hash_version,
            resume_token_version=int(payload.get("resume_token_version", 1) or 1),
            resume_token_key_id=key_id,
            resume_requested=True,
            resumed=True,
            client_supplied_run_id=bool(request.run_id),
        )

    def _decode_token(self, token: str) -> tuple[dict[str, Any], str]:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
        except ValueError as exc:
            raise ResumeTokenInvalidError("Malformed resume token") from exc

        try:
            raw_payload = self._b64decode(encoded_payload)
            payload = json.loads(raw_payload.decode("utf-8"))
            supplied_signature = self._b64decode(encoded_signature)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResumeTokenInvalidError("Malformed resume token") from exc
        if not isinstance(payload, dict):
            raise ResumeTokenInvalidError("Resume token payload must be an object")

        candidate_key_id = str(payload.get("kid") or "").strip()
        candidates = (
            [(candidate_key_id, self._keys[candidate_key_id])]
            if candidate_key_id in self._keys
            else list(self._keys.items())
        )
        for key_id, secret in candidates:
            expected_signature = hmac.new(
                secret,
                encoded_payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
            if hmac.compare_digest(expected_signature, supplied_signature):
                return payload, key_id
        raise ResumeTokenInvalidError("Resume token signature is invalid")

    def _load_keys(self, settings: Settings) -> tuple[dict[str, bytes], bool]:
        raw_keyring = str(settings.resume_token_keys_json or "").strip()
        if raw_keyring:
            try:
                parsed = json.loads(raw_keyring)
            except json.JSONDecodeError as exc:
                raise ValueError("RESUME_TOKEN_KEYS_JSON must be valid JSON") from exc
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError("RESUME_TOKEN_KEYS_JSON must be a non-empty object")
            keys = {
                str(key_id).strip(): str(secret).encode("utf-8")
                for key_id, secret in parsed.items()
                if str(key_id).strip() and str(secret)
            }
            if not keys:
                raise ValueError("RESUME_TOKEN_KEYS_JSON contains no usable keys")
            return keys, True

        configured_secret = str(settings.resume_token_secret or "").strip()
        api_key_secret = str(settings.api_key or "").strip()
        secret = configured_secret or api_key_secret
        if secret:
            return {self._active_key_id: secret.encode("utf-8")}, True
        return {self._active_key_id: secrets.token_urlsafe(48).encode("utf-8")}, False

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


class ConversationGate:
    """Fast-path rejection of overlapping turns inside one process."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, identity: RunIdentity) -> AsyncIterator[None]:
        async with self._lock:
            if identity.conversation_id in self._owners:
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


class RunRegistry:
    """Deprecated process-local idempotency registry retained for compatibility."""

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
            self._store(key, record)
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
                _RunRecord(request_hash=identity.request_hash, status="running"),
            )

    async def complete(self, identity: RunIdentity, response: ChatResponse) -> None:
        async with self._lock:
            self._store(
                (identity.conversation_id, identity.run_id),
                _RunRecord(
                    request_hash=identity.request_hash,
                    status="completed",
                    response=response.model_dump(mode="json"),
                ),
            )

    async def interrupt(self, identity: RunIdentity) -> None:
        key = (identity.conversation_id, identity.run_id)
        async with self._lock:
            stored = self._records.get(key)
            if stored is None or stored[1].status == "completed":
                return
            self._store(
                key,
                _RunRecord(request_hash=identity.request_hash, status="interrupted"),
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

    def _purge(self, now: float) -> None:
        cutoff = now - self._ttl_seconds
        for key in [
            key
            for key, (timestamp, record) in self._records.items()
            if timestamp < cutoff and record.status != "running"
        ]:
            self._records.pop(key, None)

    @staticmethod
    def _assert_same_request(record: _RunRecord, identity: RunIdentity) -> None:
        if record.request_hash != identity.request_hash:
            raise RunConflictError(
                "The supplied run_id is already bound to a different request payload"
            )
