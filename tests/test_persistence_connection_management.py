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


class _FakeNeo4jDriver:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.verify_calls: list[dict[str, Any]] = []
        self.close_calls = 0

    async def verify_connectivity(self, **kwargs: Any) -> None:
        self.verify_calls.append(kwargs)
        if self.error is not None:
            raise self.error

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_neo4j_uses_one_bounded_application_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.state.neo4j as neo4j_module

    driver = _FakeNeo4jDriver()
    create_calls: list[tuple[str, dict[str, Any]]] = []

    def create_driver(uri: str, **kwargs: Any) -> _FakeNeo4jDriver:
        create_calls.append((uri, kwargs))
        return driver

    monkeypatch.setattr(
        neo4j_module.AsyncGraphDatabase,
        "driver",
        staticmethod(create_driver),
    )
    manager = neo4j_module.Neo4jConnectionManager(
        "bolt://dbs.home.arpa:7687",
        "neo4j",
        "password",
        database="neo4j",
        max_connection_pool_size=9,
        connection_acquisition_timeout=11,
        connection_timeout=7,
        max_connection_lifetime=1800,
        keep_alive=True,
    )

    await manager.start()
    await manager.start()
    health = await manager.health()
    await manager.aclose()

    assert len(create_calls) == 1
    assert create_calls[0][0] == "bolt://dbs.home.arpa:7687"
    options = create_calls[0][1]
    assert options["auth"] == ("neo4j", "password")
    assert options["max_connection_pool_size"] == 9
    assert options["connection_acquisition_timeout"] == 11
    assert options["connection_timeout"] == 7
    assert options["max_connection_lifetime"] == 1800
    assert health["pool"]["application_scoped"] is True
    assert health["pool"]["max_connection_pool_size"] == 9
    assert driver.verify_calls == [{"database": "neo4j"}, {"database": "neo4j"}]
    assert driver.close_calls == 1


@pytest.mark.asyncio
async def test_state_runtime_reports_neo4j_connection_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.state.neo4j as neo4j_module

    driver = _FakeNeo4jDriver()
    monkeypatch.setattr(
        neo4j_module.AsyncGraphDatabase,
        "driver",
        staticmethod(lambda _uri, **_kwargs: driver),
    )
    settings = Settings(
        _env_file=None,
        state_backend="memory",
        run_repository_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
        neo4j_enabled=True,
        neo4j_uri="bolt://dbs.home.arpa:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        neo4j_max_connection_pool_size=12,
        ollama_required=False,
        mcp_required=False,
    )
    runtime = StateRuntime(settings)

    await runtime.start()
    try:
        health = await runtime.health()
        assert health["status"] == "available"
        assert health["neo4j"]["status"] == "available"
        management = health["connection_management"]["neo4j"]
        assert management["application_scoped"] is True
        assert management["pooled"] is True
        assert management["max_connection_pool_size"] == 12
    finally:
        await runtime.aclose()

    assert driver.close_calls == 1


@pytest.mark.asyncio
async def test_optional_neo4j_failure_degrades_without_discarding_other_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.state.neo4j as neo4j_module

    driver = _FakeNeo4jDriver(error=RuntimeError("neo4j unavailable"))
    monkeypatch.setattr(
        neo4j_module.AsyncGraphDatabase,
        "driver",
        staticmethod(lambda _uri, **_kwargs: driver),
    )
    settings = Settings(
        _env_file=None,
        neo4j_enabled=True,
        neo4j_uri="bolt://dbs.home.arpa:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        persistence_required=False,
        ollama_required=False,
        mcp_required=False,
    )
    runtime = StateRuntime(settings)

    await runtime.start()
    health = await runtime.health()
    await runtime.aclose()

    assert health["status"] == "degraded"
    assert health["conversation"]["backend"] == "memory"
    assert health["neo4j"]["status"] == "unavailable"
    assert driver.close_calls == 1


def test_neo4j_enabled_requires_connection_credentials() -> None:
    with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
        Settings(
            _env_file=None,
            neo4j_enabled=True,
            neo4j_uri="bolt://dbs.home.arpa:7687",
            neo4j_username="neo4j",
            neo4j_password="",
        )
