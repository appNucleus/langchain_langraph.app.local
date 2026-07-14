from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.settings import Settings
from app.state.redis import RedisConversationStore
from app.state.runtime import StateRuntime


def test_default_backends_remain_non_persistent() -> None:
    settings = Settings(
        _env_file=None,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )
    runtime = StateRuntime(settings)
    assert settings.persistence_enabled is False
    assert runtime.artifacts is None


def test_postgres_requires_database_url() -> None:
    settings = Settings(
        _env_file=None,
        state_backend="postgres",
        checkpoint_backend="memory",
        database_url="",
    )
    runtime = StateRuntime(settings)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        runtime._build_conversation_store()


@pytest.mark.asyncio
async def test_redis_store_bounds_and_expires_history() -> None:
    store = RedisConversationStore(
        "redis://example.invalid:6379/0",
        ttl_seconds=60,
        max_messages=2,
    )

    # redis.asyncio.Redis.pipeline() is a synchronous factory that returns an
    # asynchronous context manager. Mock that contract rather than making the
    # factory itself an AsyncMock/coroutine.
    pipeline = MagicMock()
    pipeline.__aenter__ = AsyncMock(return_value=pipeline)
    pipeline.__aexit__ = AsyncMock(return_value=None)
    pipeline.execute = AsyncMock(return_value=[])

    client = MagicMock()
    client.pipeline.return_value = pipeline
    store._client = client

    await store.append(
        "thread-1",
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    )

    client.pipeline.assert_called_once_with(transaction=True)
    pipeline.rpush.assert_called_once()
    pipeline.ltrim.assert_called_once_with(
        "langgraph:conversation:thread-1",
        -2,
        -1,
    )
    pipeline.expire.assert_called_once_with(
        "langgraph:conversation:thread-1",
        60,
    )
    pipeline.execute.assert_awaited_once()
