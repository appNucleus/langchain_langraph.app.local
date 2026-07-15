from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx2 as httpx
import pytest

from app.llm.ollama import OllamaClient
from app.mcp.client import MCPClient
from app.orchestration.execution_meter import (
    ExecutionBudget,
    execution_meter_scope,
    model_operation_scope,
)


class ResourceManager:
    @asynccontextmanager
    async def acquire(self, _model: str):
        yield


class MCPTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def post(self, payload, *, headers, allow_empty=False):
        del headers, allow_empty
        self.calls += 1
        if self.calls == 2:
            raise TimeoutError("provider timed out")
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"tools": []},
        }, httpx.Headers()


def ollama_settings() -> SimpleNamespace:
    return SimpleNamespace(
        ollama_connect_timeout_seconds=1,
        ollama_timeout_seconds=2,
        ollama_pool_timeout_seconds=1,
        ollama_max_connections=4,
        ollama_max_keepalive_connections=2,
        http_keepalive_expiry_seconds=30,
        ollama_base_url="http://ollama.test",
        ollama_think=False,
        ollama_keep_alive="1m",
        ollama_temperature=0.2,
        ollama_num_predict=100,
    )


def mcp_settings() -> SimpleNamespace:
    return SimpleNamespace(
        mcp_protocol_version="2025-06-18",
        mcp_client_name="test",
        mcp_client_version="1",
        mcp_enabled=True,
        mcp_session_enabled=False,
        mcp_initialize_on_startup=False,
    )


@pytest.mark.asyncio
async def test_each_ollama_http_request_is_metered_and_fallback_is_inferred() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "model",
                "message": {"content": "{}"},
                "prompt_eval_count": 10,
                "eval_count": 3,
                "load_duration": 1_000_000_000,
            },
        )

    client = OllamaClient(
        ollama_settings(),
        transport=httpx.MockTransport(handler),
        resource_manager=ResourceManager(),
    )
    budget = ExecutionBudget(60, 4, 4, 2)
    with execution_meter_scope(budget), model_operation_scope(budget):
        await client.chat(model="primary", messages=[{"role": "user", "content": "x"}])
        await client.chat(model="fallback", messages=[{"role": "user", "content": "x"}])
    await client.aclose()

    assert budget.state.physical_model_attempts == 2
    assert budget.state.fallback_attempts == 1
    assert budget.state.model_successes == 2
    assert budget.state.prompt_tokens == 20
    assert budget.state.generated_tokens == 6
    assert budget.state.model_load_seconds == 2


@pytest.mark.asyncio
async def test_each_mcp_transport_attempt_is_counted_including_timeout() -> None:
    transport = MCPTransport()
    client = MCPClient(mcp_settings(), http_transport=transport)
    budget = ExecutionBudget(60, 4, 4, 2)
    with execution_meter_scope(budget):
        result, _ = await client._send_for_session("tools/list", {}, False)
        assert result == {"tools": []}
        with pytest.raises(TimeoutError):
            await client._send_for_session("tools/list", {}, False)

    assert budget.state.tool_attempts == 2
    assert budget.state.tool_successes == 1
    assert budget.state.tool_failures == 1
    assert budget.state.tool_timeouts == 1
