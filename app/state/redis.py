from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from app.state.base import ConversationStore

_APPEND_TURN_SCRIPT = """
if redis.call('SISMEMBER', KEYS[2], ARGV[2]) == 1
   or redis.call('EXISTS', KEYS[3]) == 1 then
    redis.call('SADD', KEYS[2], ARGV[2])
    redis.call('EXPIRE', KEYS[2], ARGV[1])
    return 0
end
redis.call('SADD', KEYS[2], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[1])
redis.call('RPUSH', KEYS[1], ARGV[3], ARGV[4])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[5]), -1)
redis.call('EXPIRE', KEYS[1], ARGV[1])
return 1
"""


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

    async def append_turn(
        self,
        thread_id: str,
        *,
        run_id: str,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
    ) -> bool:
        result = await self._require_client().eval(
            _APPEND_TURN_SCRIPT,
            3,
            self._key(thread_id),
            self._turns_key(thread_id),
            self._turn_key(thread_id, run_id),
            self.ttl_seconds,
            run_id,
            json.dumps(dict(user_message), ensure_ascii=False),
            json.dumps(dict(assistant_message), ensure_ascii=False),
            self.max_messages,
        )
        return bool(result)

    async def clear(self, thread_id: str) -> None:
        client = self._require_client()
        marker_pattern = self._turn_key(thread_id, "*")
        marker_keys = [key async for key in client.scan_iter(match=marker_pattern)]
        await client.delete(
            self._key(thread_id),
            self._turns_key(thread_id),
            *marker_keys,
        )

    def _key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:{thread_id}"

    def _turns_key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:turns:{thread_id}"

    def _turn_key(self, thread_id: str, run_id: str) -> str:
        """Legacy per-run marker key retained for rolling upgrades."""

        return f"{self.key_prefix}:turn:{thread_id}:{run_id}"

    def _require_client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("Redis conversation store is not started")
        return self._client
