from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic

from app.llm.model_registry import capabilities_for
from app.observability.metrics import metrics
from app.settings import Settings


@dataclass(frozen=True)
class ResourceSnapshot:
    max_concurrency: int
    max_heavy_concurrency: int
    active: int
    active_heavy: int
    queued: int


class OllamaResourceManager:
    """Bound total Ollama pressure and serialize large-model requests."""

    def __init__(self, settings: Settings) -> None:
        self.max_concurrency = max(1, settings.ollama_max_concurrency)
        self.max_heavy_concurrency = max(1, settings.ollama_heavy_max_concurrency)
        self._all = asyncio.Semaphore(self.max_concurrency)
        self._heavy = asyncio.Semaphore(self.max_heavy_concurrency)
        self._state_lock = asyncio.Lock()
        self._active = 0
        self._active_heavy = 0
        self._queued = 0

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[None]:
        heavy = capabilities_for(model).heavy
        started_waiting = monotonic()

        async with self._state_lock:
            self._queued += 1

        acquired_all = False
        acquired_heavy = False
        counted_as_queued = True
        try:
            await self._all.acquire()
            acquired_all = True
            if heavy:
                await self._heavy.acquire()
                acquired_heavy = True

            async with self._state_lock:
                self._queued -= 1
                counted_as_queued = False
                self._active += 1
                if heavy:
                    self._active_heavy += 1

            metrics.observe("ollama.resource_wait_seconds", monotonic() - started_waiting)
            metrics.inc("ollama.resource_acquired")
            if heavy:
                metrics.inc("ollama.resource_heavy_acquired")

            yield
        except BaseException:
            # If cancellation happens while waiting for the heavy semaphore, the
            # request was still counted as queued and the total slot must be freed.
            async with self._state_lock:
                if counted_as_queued:
                    self._queued -= 1
                    counted_as_queued = False
            raise
        finally:
            if acquired_heavy:
                async with self._state_lock:
                    self._active_heavy -= 1
                self._heavy.release()
            if acquired_all:
                async with self._state_lock:
                    if self._active > 0:
                        self._active -= 1
                self._all.release()

    async def snapshot(self) -> ResourceSnapshot:
        async with self._state_lock:
            return ResourceSnapshot(
                max_concurrency=self.max_concurrency,
                max_heavy_concurrency=self.max_heavy_concurrency,
                active=self._active,
                active_heavy=self._active_heavy,
                queued=self._queued,
            )
