from __future__ import annotations

import json
from typing import Any

import asyncpg

from app.state.base import ConversationStore


class PostgresConversationStore(ConversationStore):
    """Durable conversation history backed by PostgreSQL."""

    def __init__(
        self,
        database_url: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        max_messages: int = 30,
        command_timeout: float = 30.0,
    ) -> None:
        self.database_url = database_url
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.max_messages = max_messages
        self.command_timeout = command_timeout
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
        async with self._pool.acquire() as connection:
            lock_name = "langchain-langraph-conversation-migrations"
            await connection.execute(
                "SELECT pg_advisory_lock(hashtext($1))", lock_name
            )
            try:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_conversation_messages (
                        id BIGSERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        sequence_no BIGINT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        run_id UUID,
                        message_kind TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (thread_id, sequence_no)
                    )
                    """
                )
                await connection.execute(
                    "ALTER TABLE app_conversation_messages "
                    "ADD COLUMN IF NOT EXISTS run_id UUID"
                )
                await connection.execute(
                    "ALTER TABLE app_conversation_messages "
                    "ADD COLUMN IF NOT EXISTS message_kind TEXT"
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_app_conversation_messages_thread
                    ON app_conversation_messages (thread_id, sequence_no)
                    """
                )
                await connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        uq_app_conversation_messages_run_kind
                    ON app_conversation_messages (thread_id, run_id, message_kind)
                    WHERE run_id IS NOT NULL AND message_kind IS NOT NULL
                    """
                )
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtext($1))", lock_name
                )

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health(self) -> dict[str, Any]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            value = await connection.fetchval("SELECT 1")
        return {"status": "available", "backend": "postgres", "value": value}

    async def get(self, thread_id: str) -> list[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT role, content, metadata
                FROM app_conversation_messages
                WHERE thread_id = $1
                ORDER BY sequence_no DESC
                LIMIT $2
                """,
                thread_id,
                self.max_messages,
            )
        return [
            {
                "role": row["role"],
                "content": row["content"],
                **({"metadata": dict(row["metadata"])} if row["metadata"] else {}),
            }
            for row in reversed(rows)
        ]

    async def append(self, thread_id: str, *messages: dict[str, Any]) -> None:
        if not messages:
            return
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id
                )
                next_sequence = await self._next_sequence(connection, thread_id)
                records: list[tuple[str, int, str, str, str]] = []
                for offset, message in enumerate(messages):
                    records.append(
                        (
                            thread_id,
                            int(next_sequence) + offset,
                            str(message.get("role", "assistant")),
                            str(message.get("content", "")),
                            json.dumps(message.get("metadata") or {}),
                        )
                    )
                await connection.executemany(
                    """
                    INSERT INTO app_conversation_messages
                        (thread_id, sequence_no, role, content, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    """,
                    records,
                )
                await self._trim(connection, thread_id)

    async def append_turn(
        self,
        thread_id: str,
        *,
        run_id: str,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
    ) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", thread_id
                )
                exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM app_conversation_messages
                        WHERE thread_id = $1 AND run_id = $2::uuid
                    )
                    """,
                    thread_id,
                    run_id,
                )
                if exists:
                    return False
                next_sequence = int(await self._next_sequence(connection, thread_id))
                records = []
                for offset, (kind, message) in enumerate(
                    (("user", user_message), ("assistant", assistant_message))
                ):
                    records.append(
                        (
                            thread_id,
                            next_sequence + offset,
                            str(message.get("role", kind)),
                            str(message.get("content", "")),
                            json.dumps(message.get("metadata") or {}),
                            run_id,
                            kind,
                        )
                    )
                await connection.executemany(
                    """
                    INSERT INTO app_conversation_messages (
                        thread_id, sequence_no, role, content, metadata,
                        run_id, message_kind
                    )
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6::uuid, $7)
                    ON CONFLICT DO NOTHING
                    """,
                    records,
                )
                await self._trim(connection, thread_id)
                return True

    async def clear(self, thread_id: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM app_conversation_messages WHERE thread_id = $1",
                thread_id,
            )

    @staticmethod
    async def _next_sequence(connection: Any, thread_id: str) -> int:
        return int(
            await connection.fetchval(
                """
                SELECT COALESCE(MAX(sequence_no), 0) + 1
                FROM app_conversation_messages
                WHERE thread_id = $1
                """,
                thread_id,
            )
        )

    async def _trim(self, connection: Any, thread_id: str) -> None:
        await connection.execute(
            """
            DELETE FROM app_conversation_messages
            WHERE thread_id = $1
              AND sequence_no NOT IN (
                  SELECT sequence_no
                  FROM app_conversation_messages
                  WHERE thread_id = $1
                  ORDER BY sequence_no DESC
                  LIMIT $2
              )
            """,
            thread_id,
            self.max_messages,
        )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL conversation store is not started")
        return self._pool
