from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from app.state.base import ConversationStore


class RedisConversationStore(ConversationStore):
    """Shared, TTL-based conversation history backed by Redis."""

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "langgraph:conversation",
        ttl_seconds: int = 3600,
        max_messages: int = 30,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._client: Redis | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
            )
            await self._client.ping()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> dict[str, Any]:
        value = await self._require_client().ping()
        return {"status": "available", "backend": "redis", "value": bool(value)}

    async def get(self, thread_id: str) -> list[dict[str, Any]]:
        values = await self._require_client().lrange(
            self._key(thread_id), -self.max_messages, -1
        )
        result: list[dict[str, Any]] = []
        for value in values:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                result.append(parsed)
        return result

    async def append(self, thread_id: str, *messages: dict[str, Any]) -> None:
        if not messages:
            return
        client = self._require_client()
        key = self._key(thread_id)
        encoded = [json.dumps(dict(message), ensure_ascii=False) for message in messages]
        async with client.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, *encoded)
            pipeline.ltrim(key, -self.max_messages, -1)
            pipeline.expire(key, self.ttl_seconds)
            await pipeline.execute()

    async def clear(self, thread_id: str) -> None:
        await self._require_client().delete(self._key(thread_id))

    def _key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:{thread_id}"

    def _require_client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("Redis conversation store is not started")
        return self._client
