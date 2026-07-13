from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.graph import ChatAgent
from app.settings import Settings


class FakeStateRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.conversations = object()
        self.checkpointer = object()
        self.fallback_reason: str | None = None

    async def start(self) -> None:
        if self.fail:
            raise RuntimeError("postgres unavailable")

    async def use_memory_fallback(self, reason: str) -> None:
        self.fallback_reason = reason
        self.conversations = object()
        self.checkpointer = object()


class FakeDependency:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def start(self) -> None:
        if self.fail:
            raise RuntimeError("dependency unavailable")


@pytest.mark.asyncio
async def test_optional_mcp_failure_enters_degraded_mode() -> None:
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = Settings(
        llm_backend="echo",
        mcp_enabled=True,
        mcp_required=False,
    )
    agent.state_runtime = FakeStateRuntime()
    agent.ollama = FakeDependency()
    agent.mcp = FakeDependency(fail=True)
    agent._build_graph = lambda _checkpointer: object()
    agent.startup_dependencies = {}

    await agent.start()

    assert agent.startup_dependencies["mcp"]["status"] == "unavailable"
    assert agent.startup_dependencies["mcp"]["required"] is False


@pytest.mark.asyncio
async def test_required_mcp_failure_aborts_startup() -> None:
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = Settings(
        llm_backend="echo",
        mcp_enabled=True,
        mcp_required=True,
    )
    agent.state_runtime = FakeStateRuntime()
    agent.ollama = FakeDependency()
    agent.mcp = FakeDependency(fail=True)
    agent._build_graph = lambda _checkpointer: object()
    agent.startup_dependencies = {}

    with pytest.raises(RuntimeError, match="dependency unavailable"):
        await agent.start()


@pytest.mark.asyncio
async def test_optional_persistence_failure_uses_memory_fallback() -> None:
    runtime = FakeStateRuntime(fail=True)
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = Settings(
        llm_backend="echo",
        persistence_required=False,
        artifact_backend="disabled",
    )
    agent.state_runtime = runtime
    agent.ollama = FakeDependency()
    agent.mcp = FakeDependency()
    agent._build_graph = lambda _checkpointer: object()
    agent.startup_dependencies = {}

    await agent.start()

    assert runtime.fallback_reason == "postgres unavailable"
    assert agent.startup_dependencies["persistence"]["status"] == "degraded"
