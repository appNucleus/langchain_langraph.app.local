from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.settings import Settings
from app.state.base import ConversationStore
from app.state.in_memory import BoundedInMemoryStore
from app.state.minio import MinioArtifactStore
from app.state.postgres import PostgresConversationStore
from app.state.redis import RedisConversationStore


class StateRuntime:
    """Owns Phase 4 state backends and LangGraph checkpointer lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.conversations: ConversationStore | BoundedInMemoryStore = self._memory_store()
        self.checkpointer: Any = InMemorySaver()
        self.artifacts: MinioArtifactStore | None = None
        self._checkpoint_context: AbstractAsyncContextManager[Any] | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        self.conversations = self._build_conversation_store()
        start = getattr(self.conversations, "start", None)
        if callable(start):
            await start()

        if self.settings.checkpoint_backend == "postgres":
            if not self.settings.database_url:
                raise RuntimeError("DATABASE_URL is required for PostgreSQL checkpoints")
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._checkpoint_context = AsyncPostgresSaver.from_conn_string(
                self.settings.database_url
            )
            self.checkpointer = await self._checkpoint_context.__aenter__()
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

        self._started = True

    async def aclose(self) -> None:
        if self.artifacts is not None:
            await self.artifacts.aclose()
        close = getattr(self.conversations, "aclose", None)
        if callable(close):
            await close()
        if self._checkpoint_context is not None:
            await self._checkpoint_context.__aexit__(None, None, None)
            self._checkpoint_context = None
        self._started = False

    async def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "conversation_backend": self.settings.state_backend,
            "checkpoint_backend": self.settings.checkpoint_backend,
            "artifact_backend": self.settings.artifact_backend,
        }
        health = getattr(self.conversations, "health", None)
        if callable(health):
            payload["conversation"] = await health()
        else:
            payload["conversation"] = {"status": "available", "backend": "memory"}
        if self.artifacts is not None:
            payload["artifacts"] = await self.artifacts.health()
        return payload

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
            )
        if self.settings.state_backend == "redis":
            if not self.settings.redis_url:
                raise RuntimeError("REDIS_URL is required for STATE_BACKEND=redis")
            return RedisConversationStore(
                self.settings.redis_url,
                key_prefix=self.settings.redis_key_prefix,
                ttl_seconds=self.settings.state_ttl_seconds,
                max_messages=self.settings.state_max_history_messages,
            )
        return self._memory_store()

    def _memory_store(self) -> BoundedInMemoryStore:
        return BoundedInMemoryStore(
            ttl_seconds=self.settings.state_ttl_seconds,
            max_sessions=self.settings.state_max_sessions,
            max_messages=self.settings.state_max_history_messages,
        )
