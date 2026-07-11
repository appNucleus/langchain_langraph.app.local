from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.llm.model_registry import capabilities_for
from app.settings import Settings


class OllamaResourceManager:
    """Bound local Ollama pressure and serialize very large model requests."""

    def __init__(self, settings: Settings) -> None:
        # Keep the established environment names:
        # OLLAMA_MAX_CONCURRENCY and OLLAMA_HEAVY_MAX_CONCURRENCY.
        self._all = asyncio.Semaphore(max(1, settings.ollama_max_concurrency))
        self._heavy = asyncio.Semaphore(
            max(1, settings.ollama_heavy_max_concurrency)
        )

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[None]:
        async with self._all:
            if capabilities_for(model).heavy:
                async with self._heavy:
                    yield
            else:
                yield
