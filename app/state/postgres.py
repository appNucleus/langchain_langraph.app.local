from __future__ import annotations

import json
from typing import Any

import asyncpg

from app.state.base import ConversationStore


class PostgresConversationStore(ConversationStore):
    """Durable conversation history backed by PostgreSQL.

    This store is intentionally separate from LangGraph's checkpoint tables.
    LangGraph owns checkpoint persistence; this class owns user-visible chat
    history and retention.
    """

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
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_conversation_messages (
                    id BIGSERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    sequence_no BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (thread_id, sequence_no)
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_app_conversation_messages_thread
                ON app_conversation_messages (thread_id, sequence_no)
                """
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
                next_sequence = await connection.fetchval(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1
                    FROM app_conversation_messages
                    WHERE thread_id = $1
                    """,
                    thread_id,
                )
                records: list[tuple[str, int, str, str, str]] = []
                for offset, message in enumerate(messages):
                    role = str(message.get("role", "assistant"))
                    content = str(message.get("content", ""))
                    metadata = message.get("metadata") or {}
                    records.append(
                        (
                            thread_id,
                            int(next_sequence) + offset,
                            role,
                            content,
                            json.dumps(metadata),
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

    async def clear(self, thread_id: str) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM app_conversation_messages WHERE thread_id = $1",
                thread_id,
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL conversation store is not started")
        return self._pool
