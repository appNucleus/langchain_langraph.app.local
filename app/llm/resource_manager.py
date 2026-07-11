from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from app.llm.model_registry import capabilities_for
from app.settings import Settings


class OllamaResourceManager:
    """Bounds local Ollama pressure and serializes very large model requests."""

    def __init__(self, settings: Settings) -> None:
        self._all = asyncio.Semaphore(max(1, settings.ollama_max_concurrent_requests))
        self._heavy = asyncio.Semaphore(max(1, settings.ollama_max_concurrent_heavy_requests))

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[None]:
        async with self._all:
            if capabilities_for(model).heavy:
                async with self._heavy:
                    yield
            else:
                yield
