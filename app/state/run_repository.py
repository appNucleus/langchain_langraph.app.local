from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import asyncpg

from app.orchestration.run_identity import (
    ConversationBusyError,
    RunConflictError,
    StaleLeaseError,
)
from app.schemas.run import RunIdentity

RunStatus = Literal[
    "pending",
    "running",
    "interrupted",
    "completed",
    "failed",
    "cancelled",
    "expired",
    "reconciling",
]
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
_MIGRATION_PATH = Path(__file__).with_name("migrations") / "0001_run_repository.sql"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    conversation_id: str
    execution_thread_id: str
    request_hash: str
    request_hash_version: int
    state_schema_version: int
    status: RunStatus
    fencing_token: int | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    checkpoint_id: str | None = None
    response_payload: dict[str, Any] | None = None
    termination_reason: str | None = None
    error_code: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    history_committed_at: datetime | None = None
    resume_token_version: int = 1

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True)
class RunLease:
    conversation_id: str
    run_id: str
    owner_id: str
    fencing_token: int
    expires_at: datetime


class RunRepository(ABC):
    """Durable authority for run ownership, outcomes, and replay."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...

    @abstractmethod
    async def create_or_get(self, identity: RunIdentity) -> tuple[RunRecord, bool]: ...

    @abstractmethod
    async def get(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    async def acquire(
        self,
        identity: RunIdentity,
        *,
        owner_id: str,
        ttl_seconds: int,
    ) -> RunLease: ...

    @abstractmethod
    async def renew(self, lease: RunLease, *, ttl_seconds: int) -> RunLease: ...

    @abstractmethod
    async def release(self, lease: RunLease) -> None: ...

    @abstractmethod
    async def mark_terminal(
        self,
        lease: RunLease,
        *,
        status: Literal["completed", "failed", "cancelled", "interrupted"],
        response_payload: dict[str, Any] | None,
        termination_reason: str | None,
        error_code: str | None,
        checkpoint_id: str | None = None,
    ) -> RunRecord: ...

    @abstractmethod
    async def mark_history_committed(
        self,
        lease: RunLease,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord: ...

    @abstractmethod
    async def reconcile_history_committed(
        self,
        run_id: str,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord: ...

    @abstractmethod
    async def revoke_resume_tokens(self, run_id: str) -> int: ...


class MemoryRunRepository(RunRepository):
    """Process-local repository implementing the same contract as PostgreSQL."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._leases: dict[str, RunLease] = {}
        self._fencing: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def aclose(self) -> None:
        self._started = False

    async def health(self) -> dict[str, Any]:
        return {
            "status": "available" if self._started else "unavailable",
            "backend": "memory",
            "runs": len(self._records),
            "active_leases": sum(
                lease.expires_at > _utcnow() for lease in self._leases.values()
            ),
        }

    async def create_or_get(self, identity: RunIdentity) -> tuple[RunRecord, bool]:
        async with self._lock:
            record = self._records.get(identity.run_id)
            if record is not None:
                _validate_record_identity(record, identity)
                return record, False
            now = _utcnow()
            record = RunRecord(
                run_id=identity.run_id,
                conversation_id=identity.conversation_id,
                execution_thread_id=identity.execution_thread_id,
                request_hash=identity.request_hash,
                request_hash_version=identity.request_hash_version,
                state_schema_version=identity.state_schema_version,
                status="pending",
                created_at=now,
                updated_at=now,
                resume_token_version=identity.resume_token_version,
            )
            self._records[identity.run_id] = record
            return record, True

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            return self._records.get(run_id)

    async def acquire(
        self,
        identity: RunIdentity,
        *,
        owner_id: str,
        ttl_seconds: int,
    ) -> RunLease:
        async with self._lock:
            record = self._records.get(identity.run_id)
            if record is None:
                raise RunConflictError("The run record does not exist")
            _validate_record_identity(record, identity)
            now = _utcnow()
            current = self._leases.get(identity.conversation_id)
            if current is not None and current.expires_at > now:
                raise ConversationBusyError(
                    "Another run is already active for this conversation"
                )
            token = self._fencing.get(identity.conversation_id, 0) + 1
            self._fencing[identity.conversation_id] = token
            lease = RunLease(
                conversation_id=identity.conversation_id,
                run_id=identity.run_id,
                owner_id=owner_id,
                fencing_token=token,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
            self._leases[identity.conversation_id] = lease
            self._records[identity.run_id] = replace(
                record,
                status="running",
                fencing_token=token,
                lease_owner=owner_id,
                lease_expires_at=lease.expires_at,
                started_at=record.started_at or now,
                updated_at=now,
            )
            return lease

    async def renew(self, lease: RunLease, *, ttl_seconds: int) -> RunLease:
        async with self._lock:
            self._assert_lease(lease)
            renewed = replace(
                lease,
                expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
            )
            self._leases[lease.conversation_id] = renewed
            record = self._records[lease.run_id]
            self._records[lease.run_id] = replace(
                record,
                lease_expires_at=renewed.expires_at,
                updated_at=_utcnow(),
            )
            return renewed

    async def release(self, lease: RunLease) -> None:
        async with self._lock:
            current = self._leases.get(lease.conversation_id)
            if current is not None and self._same_owner(current, lease):
                self._leases.pop(lease.conversation_id, None)

    async def mark_terminal(
        self,
        lease: RunLease,
        *,
        status: Literal["completed", "failed", "cancelled", "interrupted"],
        response_payload: dict[str, Any] | None,
        termination_reason: str | None,
        error_code: str | None,
        checkpoint_id: str | None = None,
    ) -> RunRecord:
        async with self._lock:
            self._assert_lease(lease)
            record = self._records[lease.run_id]
            now = _utcnow()
            updated = replace(
                record,
                status=status,
                response_payload=(dict(response_payload) if response_payload else None),
                termination_reason=termination_reason,
                error_code=error_code,
                checkpoint_id=checkpoint_id,
                completed_at=(now if status in _TERMINAL_STATUSES else None),
                updated_at=now,
            )
            self._records[lease.run_id] = updated
            return updated

    async def mark_history_committed(
        self,
        lease: RunLease,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord:
        async with self._lock:
            self._assert_lease(lease)
            record = self._records[lease.run_id]
            updated = replace(
                record,
                response_payload=(
                    dict(response_payload)
                    if response_payload is not None
                    else record.response_payload
                ),
                history_committed_at=_utcnow(),
                updated_at=_utcnow(),
            )
            self._records[lease.run_id] = updated
            return updated

    async def reconcile_history_committed(
        self,
        run_id: str,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord:
        async with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise RunConflictError("The run record does not exist")
            updated = replace(
                record,
                response_payload=(
                    dict(response_payload)
                    if response_payload is not None
                    else record.response_payload
                ),
                history_committed_at=record.history_committed_at or _utcnow(),
                updated_at=_utcnow(),
            )
            self._records[run_id] = updated
            return updated

    async def revoke_resume_tokens(self, run_id: str) -> int:
        async with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise RunConflictError("The run record does not exist")
            updated = replace(
                record,
                resume_token_version=record.resume_token_version + 1,
                updated_at=_utcnow(),
            )
            self._records[run_id] = updated
            return updated.resume_token_version

    def _assert_lease(self, lease: RunLease) -> None:
        current = self._leases.get(lease.conversation_id)
        if (
            current is None
            or not self._same_owner(current, lease)
            or current.expires_at <= _utcnow()
        ):
            raise StaleLeaseError("The run no longer owns the conversation lease")

    @staticmethod
    def _same_owner(current: RunLease, supplied: RunLease) -> bool:
        return (
            current.conversation_id == supplied.conversation_id
            and current.run_id == supplied.run_id
            and current.owner_id == supplied.owner_id
            and current.fencing_token == supplied.fencing_token
        )


class PostgresRunRepository(RunRepository):
    """PostgreSQL implementation with expiring leases and fencing tokens."""

    def __init__(
        self,
        database_url: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        command_timeout: float = 30.0,
        auto_setup: bool = True,
    ) -> None:
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.command_timeout = command_timeout
        self.auto_setup = auto_setup
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self.database_url,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
            command_timeout=self.command_timeout,
        )
        if self.auto_setup:
            migration = _MIGRATION_PATH.read_text(encoding="utf-8")
            async with self._pool.acquire() as connection:
                lock_name = "langchain-langraph-run-migrations"
                await connection.execute(
                    "SELECT pg_advisory_lock(hashtext($1))", lock_name
                )
                try:
                    await connection.execute(migration)
                finally:
                    await connection.execute(
                        "SELECT pg_advisory_unlock(hashtext($1))", lock_name
                    )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health(self) -> dict[str, Any]:
        async with self._require_pool().acquire() as connection:
            value = await connection.fetchval("SELECT 1")
        return {"status": "available", "backend": "postgres", "value": value}

    async def create_or_get(self, identity: RunIdentity) -> tuple[RunRecord, bool]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            inserted = await connection.fetchrow(
                """
                INSERT INTO app_runs (
                    run_id, conversation_id, execution_thread_id, request_hash,
                    request_hash_version, state_schema_version, status,
                    resume_token_version
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, 'pending', $7)
                ON CONFLICT (run_id) DO NOTHING
                RETURNING *
                """,
                identity.run_id,
                identity.conversation_id,
                identity.execution_thread_id,
                identity.request_hash,
                identity.request_hash_version,
                identity.state_schema_version,
                identity.resume_token_version,
            )
            row = inserted or await connection.fetchrow(
                "SELECT * FROM app_runs WHERE run_id = $1::uuid",
                identity.run_id,
            )
        if row is None:
            raise RuntimeError("Run insertion did not return a record")
        record = _record_from_row(row)
        _validate_record_identity(record, identity)
        return record, inserted is not None

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM app_runs WHERE run_id = $1::uuid", run_id
            )
        return _record_from_row(row) if row is not None else None

    async def acquire(
        self,
        identity: RunIdentity,
        *,
        owner_id: str,
        ttl_seconds: int,
    ) -> RunLease:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    INSERT INTO app_conversation_leases (
                        conversation_id, run_id, lease_owner, fencing_token,
                        lease_expires_at
                    )
                    VALUES (
                        $1, $2::uuid, $3::uuid, 1,
                        NOW() + ($4 * INTERVAL '1 second')
                    )
                    ON CONFLICT (conversation_id) DO UPDATE
                    SET run_id = EXCLUDED.run_id,
                        lease_owner = EXCLUDED.lease_owner,
                        fencing_token = app_conversation_leases.fencing_token + 1,
                        lease_expires_at = EXCLUDED.lease_expires_at,
                        updated_at = NOW()
                    WHERE app_conversation_leases.lease_expires_at <= NOW()
                    RETURNING conversation_id, run_id, lease_owner,
                              fencing_token, lease_expires_at
                    """,
                    identity.conversation_id,
                    identity.run_id,
                    owner_id,
                    ttl_seconds,
                )
                if row is None:
                    raise ConversationBusyError(
                        "Another run is already active for this conversation"
                    )
                updated = await connection.fetchrow(
                    """
                    UPDATE app_runs
                    SET status = 'running',
                        lease_owner = $2::uuid,
                        fencing_token = $3,
                        lease_expires_at = $4,
                        started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW()
                    WHERE run_id = $1::uuid
                    RETURNING run_id
                    """,
                    identity.run_id,
                    owner_id,
                    row["fencing_token"],
                    row["lease_expires_at"],
                )
                if updated is None:
                    raise RunConflictError("The run record does not exist")
        return RunLease(
            conversation_id=str(row["conversation_id"]),
            run_id=str(row["run_id"]),
            owner_id=str(row["lease_owner"]),
            fencing_token=int(row["fencing_token"]),
            expires_at=row["lease_expires_at"],
        )

    async def renew(self, lease: RunLease, *, ttl_seconds: int) -> RunLease:
        async with self._require_pool().acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    UPDATE app_conversation_leases
                    SET lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
                        updated_at = NOW()
                    WHERE conversation_id = $1
                      AND run_id = $2::uuid
                      AND lease_owner = $3::uuid
                      AND fencing_token = $4
                      AND lease_expires_at > NOW()
                    RETURNING lease_expires_at
                    """,
                    lease.conversation_id,
                    lease.run_id,
                    lease.owner_id,
                    lease.fencing_token,
                    ttl_seconds,
                )
                if row is None:
                    raise StaleLeaseError(
                        "The run no longer owns the conversation lease"
                    )
                await connection.execute(
                    """
                    UPDATE app_runs
                    SET lease_expires_at = $2, updated_at = NOW()
                    WHERE run_id = $1::uuid AND fencing_token = $3
                    """,
                    lease.run_id,
                    row["lease_expires_at"],
                    lease.fencing_token,
                )
        return replace(lease, expires_at=row["lease_expires_at"])

    async def release(self, lease: RunLease) -> None:
        async with self._require_pool().acquire() as connection:
            await connection.execute(
                """
                UPDATE app_conversation_leases
                SET lease_expires_at = NOW(), updated_at = NOW()
                WHERE conversation_id = $1
                  AND run_id = $2::uuid
                  AND lease_owner = $3::uuid
                  AND fencing_token = $4
                """,
                lease.conversation_id,
                lease.run_id,
                lease.owner_id,
                lease.fencing_token,
            )

    async def mark_terminal(
        self,
        lease: RunLease,
        *,
        status: Literal["completed", "failed", "cancelled", "interrupted"],
        response_payload: dict[str, Any] | None,
        termination_reason: str | None,
        error_code: str | None,
        checkpoint_id: str | None = None,
    ) -> RunRecord:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE app_runs
                SET status = $4,
                    response_payload = $5::jsonb,
                    termination_reason = $6,
                    error_code = $7,
                    checkpoint_id = $8,
                    completed_at = CASE
                        WHEN $4 IN ('completed', 'failed', 'cancelled') THEN NOW()
                        ELSE NULL
                    END,
                    updated_at = NOW()
                WHERE run_id = $1::uuid
                  AND lease_owner = $2::uuid
                  AND fencing_token = $3
                  AND EXISTS (
                      SELECT 1 FROM app_conversation_leases lease
                      WHERE lease.conversation_id = app_runs.conversation_id
                        AND lease.run_id = app_runs.run_id
                        AND lease.lease_owner = $2::uuid
                        AND lease.fencing_token = $3
                        AND lease.lease_expires_at > NOW()
                  )
                RETURNING *
                """,
                lease.run_id,
                lease.owner_id,
                lease.fencing_token,
                status,
                json.dumps(response_payload) if response_payload is not None else None,
                termination_reason,
                error_code,
                checkpoint_id,
            )
        if row is None:
            raise StaleLeaseError("A stale run owner cannot persist an outcome")
        return _record_from_row(row)

    async def mark_history_committed(
        self,
        lease: RunLease,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE app_runs
                SET history_committed_at = COALESCE(history_committed_at, NOW()),
                    response_payload = COALESCE($4::jsonb, response_payload),
                    updated_at = NOW()
                WHERE run_id = $1::uuid
                  AND lease_owner = $2::uuid
                  AND fencing_token = $3
                  AND EXISTS (
                      SELECT 1 FROM app_conversation_leases lease
                      WHERE lease.conversation_id = app_runs.conversation_id
                        AND lease.run_id = app_runs.run_id
                        AND lease.lease_owner = $2::uuid
                        AND lease.fencing_token = $3
                        AND lease.lease_expires_at > NOW()
                  )
                RETURNING *
                """,
                lease.run_id,
                lease.owner_id,
                lease.fencing_token,
                json.dumps(response_payload) if response_payload is not None else None,
            )
        if row is None:
            raise StaleLeaseError("A stale run owner cannot commit conversation history")
        return _record_from_row(row)

    async def reconcile_history_committed(
        self,
        run_id: str,
        *,
        response_payload: dict[str, Any] | None = None,
    ) -> RunRecord:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE app_runs
                SET history_committed_at = COALESCE(history_committed_at, NOW()),
                    response_payload = COALESCE($2::jsonb, response_payload),
                    updated_at = NOW()
                WHERE run_id = $1::uuid AND status IN ('completed', 'failed', 'cancelled')
                RETURNING *
                """,
                run_id,
                json.dumps(response_payload) if response_payload is not None else None,
            )
        if row is None:
            raise RunConflictError("The terminal run record does not exist")
        return _record_from_row(row)

    async def revoke_resume_tokens(self, run_id: str) -> int:
        async with self._require_pool().acquire() as connection:
            value = await connection.fetchval(
                """
                UPDATE app_runs
                SET resume_token_version = resume_token_version + 1,
                    updated_at = NOW()
                WHERE run_id = $1::uuid
                RETURNING resume_token_version
                """,
                run_id,
            )
        if value is None:
            raise RunConflictError("The run record does not exist")
        return int(value)

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL run repository is not started")
        return self._pool


def _validate_record_identity(record: RunRecord, identity: RunIdentity) -> None:
    if record.conversation_id != identity.conversation_id:
        raise RunConflictError("The run_id belongs to a different conversation")
    if record.execution_thread_id != identity.execution_thread_id:
        raise RunConflictError("The run_id has a different execution identity")
    if (
        record.request_hash != identity.request_hash
        or record.request_hash_version != identity.request_hash_version
    ):
        raise RunConflictError(
            "The supplied run_id is already bound to a different request payload"
        )
    if record.state_schema_version != identity.state_schema_version:
        raise RunConflictError("The run_id uses an incompatible state schema")


def _record_from_row(row: Any) -> RunRecord:
    response = row["response_payload"]
    if isinstance(response, str):
        response = json.loads(response)
    return RunRecord(
        run_id=str(row["run_id"]),
        conversation_id=str(row["conversation_id"]),
        execution_thread_id=str(row["execution_thread_id"]),
        request_hash=str(row["request_hash"]),
        request_hash_version=int(row["request_hash_version"]),
        state_schema_version=int(row["state_schema_version"]),
        status=str(row["status"]),
        fencing_token=(
            int(row["fencing_token"]) if row["fencing_token"] is not None else None
        ),
        lease_owner=(str(row["lease_owner"]) if row["lease_owner"] else None),
        lease_expires_at=row["lease_expires_at"],
        checkpoint_id=row["checkpoint_id"],
        response_payload=(dict(response) if response is not None else None),
        termination_reason=row["termination_reason"],
        error_code=row["error_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        history_committed_at=row["history_committed_at"],
        resume_token_version=int(row["resume_token_version"]),
    )
