from __future__ import annotations

import asyncio
from collections import OrderedDict
from time import monotonic
from typing import Any

from app.observability.metrics import metrics


class BoundedInMemoryStore:
    """TTL- and LRU-bounded process-local conversation history."""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_sessions: int,
        max_messages: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")

        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self._data: OrderedDict[
            str,
            tuple[float, list[dict[str, Any]]],
        ] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> list[dict[str, Any]]:
        if not key:
            return []
        async with self._lock:
            now = monotonic()
            self._purge(now)
            item = self._data.get(key)
            if item is None:
                metrics.inc("state.memory_miss")
                return []

            _, messages = item
            # TTL represents inactivity. A successful read keeps an actively used
            # conversation alive and also updates its LRU position.
            self._data[key] = (now, messages)
            self._data.move_to_end(key)
            metrics.inc("state.memory_hit")
            return [dict(message) for message in messages]

    async def append(self, key: str, *messages: dict[str, Any]) -> None:
        if not key:
            raise ValueError("conversation key must not be empty")
        if not messages:
            return
        if any(not isinstance(message, dict) for message in messages):
            raise TypeError("conversation messages must be dictionaries")

        async with self._lock:
            now = monotonic()
            self._purge(now)
            current = self._data.get(key, (0.0, []))[1]
            updated = (current + [dict(message) for message in messages])[
                -self.max_messages :
            ]
            self._data[key] = (now, updated)
            self._data.move_to_end(key)

            evicted = 0
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)
                evicted += 1
            if evicted:
                metrics.inc("state.memory_lru_evictions", evicted)
            metrics.inc("state.memory_appends", len(messages))

    async def clear(self, key: str | None = None) -> None:
        async with self._lock:
            if key is None:
                removed = len(self._data)
                self._data.clear()
            else:
                removed = int(self._data.pop(key, None) is not None)
            if removed:
                metrics.inc("state.memory_cleared_sessions", removed)

    async def health(self) -> dict[str, Any]:
        async with self._lock:
            now = monotonic()
            self._purge(now)
            return {
                "status": "available",
                "backend": "memory",
                "sessions": len(self._data),
                "max_sessions": self.max_sessions,
                "max_messages_per_session": self.max_messages,
                "ttl_seconds": self.ttl,
            }

    def _purge(self, now: float) -> None:
        cutoff = now - self.ttl
        expired = [key for key, (timestamp, _) in self._data.items() if timestamp < cutoff]
        for key in expired:
            self._data.pop(key, None)
        if expired:
            metrics.inc("state.memory_ttl_evictions", len(expired))
