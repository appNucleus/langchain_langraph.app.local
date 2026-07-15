from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.logging_config import log_kv
from app.settings import Settings
from app.state.base import ConversationStore
from app.state.in_memory import BoundedInMemoryStore
from app.state.minio import MinioArtifactStore
from app.state.neo4j import Neo4jConnectionManager
from app.state.postgres import PostgresConversationStore
from app.state.redis import RedisConversationStore
from app.state.run_repository import (
    MemoryRunRepository,
    PostgresRunRepository,
    RunRepository,
)

logger = logging.getLogger(__name__)


def _checkpoint_serializer() -> JsonPlusSerializer:
    """Create the strict serializer used by every checkpoint backend."""

    return JsonPlusSerializer(
        allowed_msgpack_modules=(("app.schemas.execution", "ExecutionBudget"),)
    )


class StateRuntime:
    """Own conversation, run, checkpoint, and artifact backend lifecycles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conversations: ConversationStore | BoundedInMemoryStore = self._memory_store()
        self.runs: RunRepository = MemoryRunRepository()
        self.checkpointer: Any = InMemorySaver(serde=_checkpoint_serializer())
        self.artifacts: MinioArtifactStore | None = None
        self.neo4j: Neo4jConnectionManager | None = None
        self._checkpoint_pool: AsyncConnectionPool | None = None
        self._postgres_pool: asyncpg.Pool | None = None
        self._started = False
        self.degraded_reason: str | None = None
        self.neo4j_degraded_reason: str | None = None

    async def start(self) -> None:
        if self._started:
            return
        try:
            await self._start_postgres_pool()

            self.conversations = self._build_conversation_store()
            start = getattr(self.conversations, "start", None)
            if callable(start):
                await start()

            self.runs = self._build_run_repository()
            await self.runs.start()

            if self.settings.checkpoint_backend == "postgres":
                if not self.settings.database_url:
                    raise RuntimeError("DATABASE_URL is required for PostgreSQL checkpoints")
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # LangGraph's checkpointer uses psycopg while the application stores use
                # asyncpg. The drivers cannot share one physical pool, so each driver has
                # one bounded application-scoped pool instead of per-request connections.
                self._checkpoint_pool = AsyncConnectionPool(
                    conninfo=self.settings.database_url,
                    min_size=self.settings.postgres_pool_min_size,
                    max_size=self.settings.postgres_pool_max_size,
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                    },
                    open=False,
                    timeout=self.settings.postgres_command_timeout_seconds,
                )
                await self._checkpoint_pool.open(wait=True)
                self.checkpointer = AsyncPostgresSaver(
                    self._checkpoint_pool,
                    serde=_checkpoint_serializer(),
                )
                if self.settings.postgres_auto_setup:
                    await self.checkpointer.setup()

            if self.settings.artifact_backend == "minio":
                self.artifacts = MinioArtifactStore(
                    self.settings.minio_endpoint,
                    self.settings.minio_access_key,
                    self.settings.minio_secret_key,
                    bucket=self.settings.minio_bucket,
                    secure=self.settings.minio_secure,
                )
                await self.artifacts.start()

            await self._start_neo4j()

            self._started = True
            self.degraded_reason = None
        except Exception:
            await self.aclose()
            raise

    async def use_memory_fallback(self, reason: str) -> None:
        """Switch optional persistence failures to a clean in-memory runtime."""

        await self.aclose()
        self.conversations = self._memory_store()
        start = getattr(self.conversations, "start", None)
        if callable(start):
            await start()
        self.runs = MemoryRunRepository()
        await self.runs.start()
        self.checkpointer = InMemorySaver(serde=_checkpoint_serializer())
        self.artifacts = None
        self.neo4j = None
        self._checkpoint_pool = None
        self._started = True
        self.degraded_reason = reason
        self.neo4j_degraded_reason = None

    async def aclose(self) -> None:
        if self.neo4j is not None:
            try:
                await self.neo4j.aclose()
            except Exception:  # noqa: BLE001 - shutdown remains best effort
                logger.exception("neo4j_driver_close_failed")
            finally:
                self.neo4j = None

        if self.artifacts is not None:
            try:
                await self.artifacts.aclose()
            finally:
                self.artifacts = None

        try:
            await self.runs.aclose()
        except Exception:  # noqa: BLE001 - shutdown remains best effort
            logger.exception("run_repository_close_failed")

        close = getattr(self.conversations, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:  # noqa: BLE001 - shutdown remains best effort
                logger.exception("conversation_store_close_failed")

        if self._checkpoint_pool is not None:
            try:
                await self._checkpoint_pool.close()
            except Exception:  # noqa: BLE001 - shutdown remains best effort
                logger.exception("checkpoint_pool_close_failed")
            finally:
                self._checkpoint_pool = None

        if self._postgres_pool is not None:
            try:
                await self._postgres_pool.close()
            except Exception:  # noqa: BLE001 - shutdown remains best effort
                logger.exception("postgres_pool_close_failed")
            finally:
                self._postgres_pool = None

        self._started = False

    async def _checkpoint_health(self) -> dict[str, Any]:
        config = {
            "configurable": {
                "thread_id": "__health_check__",
                "checkpoint_ns": "",
            }
        }
        timeout = self.settings.persistence_health_timeout_seconds
        try:
            async_get = getattr(self.checkpointer, "aget_tuple", None)
            if callable(async_get):
                await asyncio.wait_for(async_get(config), timeout=timeout)
            else:
                get = getattr(self.checkpointer, "get_tuple", None)
                if callable(get):
                    await asyncio.wait_for(
                        asyncio.to_thread(get, config), timeout=timeout
                    )
                else:
                    raise RuntimeError("checkpointer does not expose a health-readable API")
        except Exception as exc:
            log_kv(
                logger,
                logging.ERROR,
                "checkpoint_health_failed",
                backend=self.settings.checkpoint_backend,
                error_type=type(exc).__name__,
                error=str(exc).strip() or "health check failed",
            )
            return {
                "status": "unavailable",
                "backend": self.settings.checkpoint_backend,
                "error": (
                    f"{type(exc).__name__}: {str(exc).strip() or 'health check failed'}"
                    if self.settings.expose_internal_health_details
                    else "Dependency unavailable."
                ),
            }
        return {
            "status": "available",
            "backend": self.settings.checkpoint_backend,
        }

    async def health(self) -> dict[str, Any]:
        degraded = bool(self.degraded_reason or self.neo4j_degraded_reason)
        payload: dict[str, Any] = {
            "status": "degraded" if degraded else "available",
            "conversation_backend": self.settings.state_backend,
            "run_repository_backend": self.settings.run_repository_backend,
            "checkpoint_backend": self.settings.checkpoint_backend,
            "artifact_backend": self.settings.artifact_backend,
            "connection_management": self._connection_management_status(),
        }
        if self.degraded_reason:
            payload["degraded_reason"] = (
                self.degraded_reason
                if self.settings.expose_internal_health_details
                else "Optional persistence dependency unavailable; using memory fallback."
            )
            payload["effective_conversation_backend"] = "memory"
            payload["effective_run_repository_backend"] = "memory"
            payload["effective_checkpoint_backend"] = "memory"

        health = getattr(self.conversations, "health", None)
        payload["conversation"] = (
            await health()
            if callable(health)
            else {"status": "available", "backend": "memory"}
        )
        try:
            payload["runs"] = await asyncio.wait_for(
                self.runs.health(),
                timeout=self.settings.persistence_health_timeout_seconds,
            )
        except Exception as exc:
            payload["runs"] = {
                "status": "unavailable",
                "backend": self.settings.run_repository_backend,
                "error": (
                    f"{type(exc).__name__}: {str(exc).strip()}"
                    if self.settings.expose_internal_health_details
                    else "Dependency unavailable."
                ),
            }
        payload["checkpoint"] = await self._checkpoint_health()

        if self.artifacts is not None:
            payload["artifacts"] = await self.artifacts.health()
        elif self.settings.artifact_backend == "minio":
            payload["artifacts"] = {
                "status": "unavailable",
                "backend": "minio",
                "error": (
                    self.degraded_reason or "artifact backend not started"
                    if self.settings.expose_internal_health_details
                    else "Dependency unavailable."
                ),
            }
        else:
            payload["artifacts"] = {"status": "disabled", "backend": "disabled"}

        if self.neo4j is not None:
            try:
                payload["neo4j"] = await asyncio.wait_for(
                    self.neo4j.health(),
                    timeout=self.settings.persistence_health_timeout_seconds,
                )
            except Exception as exc:
                payload["neo4j"] = {
                    "status": "unavailable",
                    "backend": "neo4j",
                    "error": (
                        f"{type(exc).__name__}: {str(exc).strip()}"
                        if self.settings.expose_internal_health_details
                        else "Dependency unavailable."
                    ),
                }
        elif self.settings.neo4j_enabled:
            payload["neo4j"] = {
                "status": "unavailable",
                "backend": "neo4j",
                "error": (
                    self.neo4j_degraded_reason or "Neo4j driver not started"
                    if self.settings.expose_internal_health_details
                    else "Dependency unavailable."
                ),
            }
        else:
            payload["neo4j"] = {"status": "disabled", "backend": "neo4j"}

        return payload

    async def _start_neo4j(self) -> None:
        if not self.settings.neo4j_enabled or self.neo4j is not None:
            return
        manager = Neo4jConnectionManager(
            self.settings.neo4j_uri,
            self.settings.neo4j_username,
            self.settings.neo4j_password,
            database=self.settings.neo4j_database,
            max_connection_pool_size=self.settings.neo4j_max_connection_pool_size,
            connection_acquisition_timeout=(
                self.settings.neo4j_connection_acquisition_timeout_seconds
            ),
            connection_timeout=self.settings.neo4j_connection_timeout_seconds,
            max_connection_lifetime=(
                self.settings.neo4j_max_connection_lifetime_seconds
            ),
            keep_alive=self.settings.neo4j_keep_alive,
        )
        try:
            await manager.start()
        except Exception as exc:
            await manager.aclose()
            if self.settings.persistence_required:
                raise
            self.neo4j_degraded_reason = (
                f"{type(exc).__name__}: {str(exc).strip() or 'startup failed'}"
            )
            log_kv(
                logger,
                logging.ERROR,
                "neo4j_startup_degraded",
                error_type=type(exc).__name__,
                error=str(exc).strip() or "startup failed",
            )
            return
        self.neo4j = manager
        self.neo4j_degraded_reason = None

    async def _start_postgres_pool(self) -> None:
        uses_asyncpg = any(
            (
                self.settings.state_backend == "postgres",
                self.settings.run_repository_backend == "postgres",
            )
        )
        if not uses_asyncpg or self._postgres_pool is not None:
            return
        if not self.settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is required for PostgreSQL conversation or run storage"
            )
        self._postgres_pool = await asyncpg.create_pool(
            dsn=self.settings.database_url,
            min_size=self.settings.postgres_pool_min_size,
            max_size=self.settings.postgres_pool_max_size,
            command_timeout=self.settings.postgres_command_timeout_seconds,
        )

    def _connection_management_status(self) -> dict[str, Any]:
        pool = self._postgres_pool
        postgres: dict[str, Any]
        if pool is None:
            postgres = {"status": "disabled", "shared_asyncpg_pool": False}
        else:
            postgres = {
                "status": "available",
                "shared_asyncpg_pool": True,
                "size": pool.get_size(),
                "idle": pool.get_idle_size(),
                "min_size": pool.get_min_size(),
                "max_size": pool.get_max_size(),
            }
        return {
            "postgres": postgres,
            "checkpoint": {
                "driver": "psycopg",
                "application_scoped": self._checkpoint_pool is not None,
                "pooled": self._checkpoint_pool is not None,
                "min_size": (
                    self.settings.postgres_pool_min_size
                    if self._checkpoint_pool is not None
                    else 0
                ),
                "max_size": (
                    self.settings.postgres_pool_max_size
                    if self._checkpoint_pool is not None
                    else 0
                ),
            },
            "redis": {
                "application_scoped": self.settings.state_backend == "redis",
            },
            "minio": {
                "application_scoped": self.settings.artifact_backend == "minio",
            },
            "neo4j": (
                self.neo4j.connection_status()
                if self.neo4j is not None
                else {
                    "application_scoped": False,
                    "pooled": False,
                    "max_connection_pool_size": (
                        self.settings.neo4j_max_connection_pool_size
                        if self.settings.neo4j_enabled
                        else 0
                    ),
                }
            ),
        }

    def _build_conversation_store(self) -> ConversationStore | BoundedInMemoryStore:
        if self.settings.state_backend == "postgres":
            if not self.settings.database_url:
                raise RuntimeError("DATABASE_URL is required for STATE_BACKEND=postgres")
            return PostgresConversationStore(
                self.settings.database_url,
                min_pool_size=self.settings.postgres_pool_min_size,
                max_pool_size=self.settings.postgres_pool_max_size,
                max_messages=self.settings.state_max_history_messages,
                command_timeout=self.settings.postgres_command_timeout_seconds,
                pool=self._postgres_pool,
            )
        if self.settings.state_backend == "redis":
            if not self.settings.redis_url:
                raise RuntimeError("REDIS_URL is required for STATE_BACKEND=redis")
            return RedisConversationStore(
                self.settings.redis_url,
                key_prefix=self.settings.redis_key_prefix,
                ttl_seconds=self.settings.state_ttl_seconds,
                max_messages=self.settings.state_max_history_messages,
                max_connections=self.settings.redis_max_connections,
            )
        return self._memory_store()

    def _build_run_repository(self) -> RunRepository:
        if self.settings.run_repository_backend == "postgres":
            if not self.settings.database_url:
                raise RuntimeError(
                    "DATABASE_URL is required for RUN_REPOSITORY_BACKEND=postgres"
                )
            return PostgresRunRepository(
                self.settings.database_url,
                min_pool_size=self.settings.postgres_pool_min_size,
                max_pool_size=self.settings.postgres_pool_max_size,
                command_timeout=self.settings.postgres_command_timeout_seconds,
                auto_setup=self.settings.postgres_auto_setup,
                pool=self._postgres_pool,
            )
        return MemoryRunRepository()

    def _memory_store(self) -> BoundedInMemoryStore:
        return BoundedInMemoryStore(
            ttl_seconds=self.settings.state_ttl_seconds,
            max_sessions=self.settings.state_max_sessions,
            max_messages=self.settings.state_max_history_messages,
        )
