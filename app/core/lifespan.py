from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app import __version__
from app.logging_config import log_kv
from app.settings import Settings

logger = logging.getLogger("app.factory")


def build_lifespan(
    settings: Settings,
    chat_agent: Any,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the application lifespan without creating runtime dependencies."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log_kv(
            logger,
            logging.INFO,
            "app_start",
            version=__version__,
            environment=settings.environment,
            backend=settings.llm_backend,
            state_backend=settings.state_backend,
            run_repository_backend=settings.run_repository_backend,
            checkpoint_backend=settings.checkpoint_backend,
            artifact_backend=settings.artifact_backend,
        )
        try:
            start = getattr(chat_agent, "start", None)
            if callable(start):
                await start()
            yield
        finally:
            close = getattr(chat_agent, "aclose", None)
            if callable(close):
                await close()
            log_kv(logger, logging.INFO, "app_stop", version=__version__)

    return lifespan
