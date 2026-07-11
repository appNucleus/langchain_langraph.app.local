from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from typing import Any


class InMemoryMetrics:
    """Lightweight process-local metrics; no database or external exporter required."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._counts: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = defaultdict(list)

    async def increment(self, name: str, amount: int = 1) -> None:
        async with self._lock:
            self._counts[name] += amount

    async def observe(self, name: str, seconds: float) -> None:
        async with self._lock:
            values = self._durations[name]
            values.append(seconds)
            if len(values) > 500:
                del values[:-500]

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            durations = {
                key: {
                    "count": len(values),
                    "average_seconds": (sum(values) / len(values)) if values else 0.0,
                    "max_seconds": max(values) if values else 0.0,
                }
                for key, values in self._durations.items()
            }
            return {"counts": dict(self._counts), "durations": durations}
