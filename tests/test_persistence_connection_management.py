from __future__ import annotations

from typing import Any

import pytest

import app.state.redis as redis_module
import app.state.runtime as runtime_module
from app.settings import Settings
from app.state.postgres import PostgresConversationStore
from app.state.redis import RedisConversationStore
from app.state.run_repository import PostgresRunRepository
from app.state.runtime import StateRuntime


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *arguments: Any) -> str:
        self.executed.append((query, arguments))
        return "OK"

    async def fetchval(self, query: str, *arguments: Any) -> int:
        self.executed.append((query, arguments))
        return 1


class _AcquireContext:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self.connection

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.connection = _FakeConnection()
        self.close_calls = 0

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)

    async def close(self) -> None:
        self.close_calls += 1

    def get_size(self) -> int:
        return 2

    def get_idle_size(self) -> int:
        return 2

    def get_min_size(self) -> int:
        return 1

    def get_max_size(self) -> int:
        return 10


@pytest.mark.asyncio
async def test_state_runtime_reuses_one_asyncpg_pool_for_postgres_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    create_calls: list[dict[str, Any]] = []

    async def create_pool(**kwargs: Any) -> _FakePool:
        create_calls.append(kwargs)
        return pool

    monkeypatch.setattr(runtime_module.asyncpg, "create_pool", create_pool)
    settings = Settings(
        _env_file=None,
        state_backend="postgres",
        run_repository_backend="postgres",
        checkpoint_backend="memory",
        artifact_backend="disabled",
        database_url="postgresql://user:password@dbs.home.arpa:5432/langgraph_app",
        resume_token_secret="test-only-signing-key",
        ollama_required=False,
        mcp_required=False,
    )
    runtime = StateRuntime(settings)

    await runtime.start()
    try:
        assert len(create_calls) == 1
        assert isinstance(runtime.conversations, PostgresConversationStore)
        assert isinstance(runtime.runs, PostgresRunRepository)
        assert runtime.conversations._pool is pool
        assert runtime.runs._pool is pool

        health = await runtime.health()
        management = health["connection_management"]["postgres"]
        assert management["shared_asyncpg_pool"] is True
        assert management["size"] == 2
        assert health["conversation"]["pool"]["shared"] is True
        assert health["runs"]["pool"]["shared"] is True
    finally:
        await runtime.aclose()

    assert pool.close_calls == 1


@pytest.mark.asyncio
async def test_external_postgres_pool_is_not_closed_by_individual_consumers() -> None:
    pool = _FakePool()
    conversation = PostgresConversationStore(
        "postgresql://unused",
        pool=pool,  # type: ignore[arg-type]
    )
    runs = PostgresRunRepository(
        "postgresql://unused",
        pool=pool,  # type: ignore[arg-type]
    )

    await conversation.start()
    await runs.start()
    await conversation.aclose()
    await runs.aclose()

    assert pool.close_calls == 0


class _FakeRedisClient:
    def __init__(self) -> None:
        self.closed = False

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_redis_uses_one_bounded_application_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeRedisClient()
    captured: dict[str, Any] = {}

    def from_url(url: str, **kwargs: Any) -> _FakeRedisClient:
        captured["url"] = url
        captured.update(kwargs)
        return client

    monkeypatch.setattr(redis_module.Redis, "from_url", staticmethod(from_url))
    store = RedisConversationStore(
        "redis://default:password@dbs.home.arpa:6379/0",
        max_connections=7,
    )

    await store.start()
    await store.start()
    health = await store.health()
    await store.aclose()

    assert captured["max_connections"] == 7
    assert captured["health_check_interval"] == 30
    assert health["pool"]["application_scoped"] is True
    assert health["pool"]["max_connections"] == 7
    assert client.closed is True


def test_postgres_pool_minimum_cannot_exceed_maximum() -> None:
    with pytest.raises(ValueError, match="POSTGRES_POOL_MIN_SIZE"):
        Settings(
            _env_file=None,
            postgres_pool_min_size=5,
            postgres_pool_max_size=4,
        )

class _FakeCheckpointPool:
    instances: list["_FakeCheckpointPool"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.open_calls: list[bool] = []
        self.close_calls = 0
        self.__class__.instances.append(self)

    async def open(self, *, wait: bool = False) -> None:
        self.open_calls.append(wait)

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_postgres_checkpointer_uses_one_bounded_application_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCheckpointPool.instances.clear()
    monkeypatch.setattr(runtime_module, "AsyncConnectionPool", _FakeCheckpointPool)
    settings = Settings(
        _env_file=None,
        state_backend="memory",
        run_repository_backend="memory",
        checkpoint_backend="postgres",
        artifact_backend="disabled",
        database_url="postgresql://user:password@dbs.home.arpa:5432/langgraph_app",
        postgres_pool_min_size=2,
        postgres_pool_max_size=6,
        postgres_auto_setup=False,
        ollama_required=False,
        mcp_required=False,
    )
    runtime = StateRuntime(settings)

    await runtime.start()
    pool = _FakeCheckpointPool.instances[0]
    try:
        management = runtime._connection_management_status()["checkpoint"]
        assert len(_FakeCheckpointPool.instances) == 1
        assert pool.open_calls == [True]
        assert pool.kwargs["min_size"] == 2
        assert pool.kwargs["max_size"] == 6
        assert management["application_scoped"] is True
        assert management["pooled"] is True
        assert management["min_size"] == 2
        assert management["max_size"] == 6
    finally:
        await runtime.aclose()

    assert pool.close_calls == 1
