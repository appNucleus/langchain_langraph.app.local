from __future__ import annotations

import asyncio
import json

import httpx2 as httpx
import pytest
from pydantic import BaseModel

from app.agents.base import StructuredAgent
from app.llm.ollama import OllamaClient
from app.llm.resource_manager import OllamaResourceManager
from app.settings import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_backend": "ollama",
        "ollama_base_url": "http://ollama.test",
        "ollama_max_concurrency": 2,
        "ollama_heavy_max_concurrency": 1,
        "inventory_cache_ttl_seconds": 60,
        "inventory_stale_if_error_seconds": 300,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_http_client_is_reused_and_model_inventory_is_cached() -> None:
    tags_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tags_calls
        if request.url.path == "/api/tags":
            tags_calls += 1
            return httpx.Response(
                200,
                json={"models": [{"name": f"model-{tags_calls}:1b"}]},
            )
        raise AssertionError(request.url.path)

    client = OllamaClient(_settings(), transport=httpx.MockTransport(handler))
    http_client_1 = await client._get_client()
    http_client_2 = await client._get_client()
    first = await client.list_models()
    second = await client.list_models()
    refreshed = await client.list_models(force_refresh=True)

    assert http_client_1 is http_client_2
    assert first == second == [{"name": "model-1:1b"}]
    assert refreshed == [{"name": "model-2:1b"}]
    assert tags_calls == 2
    await client.aclose()


def test_default_clients_share_one_runtime_and_resource_manager() -> None:
    settings = _settings()
    first = OllamaClient(settings)
    second = OllamaClient(settings)

    assert first.runtime_identity == second.runtime_identity
    assert first.resources is second.resources


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float


@pytest.mark.asyncio
async def test_structured_agent_uses_ollama_json_schema_and_shared_client() -> None:
    observed_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_payload
        assert request.url.path == "/api/chat"
        observed_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "qwen:4b",
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"yes","confidence":0.9}',
                },
                "done": True,
                "load_duration": 10,
                "eval_count": 3,
            },
        )

    settings = _settings(ollama_model="qwen:4b")
    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    agent = StructuredAgent(settings, ollama_client=client)
    result = await agent.invoke_json(
        system="Return a structured answer.",
        payload={"question": "ok?"},
        schema=StructuredAnswer,
    )

    assert result.answer == "yes"
    assert observed_payload["format"] == StructuredAnswer.model_json_schema()
    assert observed_payload["stream"] is False
    assert observed_payload["keep_alive"] == settings.ollama_keep_alive
    assert observed_payload["options"]["temperature"] == 0.0  # type: ignore[index]
    assert observed_payload["options"]["num_ctx"] == settings.ollama_num_ctx  # type: ignore[index]
    await client.aclose()


@pytest.mark.asyncio
async def test_native_stream_parses_ndjson_tokens() -> None:
    body = "\n".join(
        [
            json.dumps({"message": {"content": "hello "}, "done": False}),
            json.dumps(
                {
                    "message": {"content": "world"},
                    "done": True,
                    "eval_count": 2,
                    "eval_duration": 1_000_000_000,
                }
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, text=body)

    client = OllamaClient(_settings(), transport=httpx.MockTransport(handler))
    tokens = [
        token
        async for token in client.stream_chat(
            model="qwen:4b",
            messages=[{"role": "user", "content": "hello"}],
        )
    ]
    assert tokens == ["hello ", "world"]
    await client.aclose()


@pytest.mark.asyncio
async def test_global_and_heavy_concurrency_limits_are_enforced() -> None:
    settings = _settings(
        ollama_max_concurrency=2,
        ollama_heavy_max_concurrency=1,
    )
    manager = OllamaResourceManager(settings)
    active = 0
    active_heavy = 0
    peak = 0
    peak_heavy = 0
    lock = asyncio.Lock()

    async def work(model: str) -> None:
        nonlocal active, active_heavy, peak, peak_heavy
        heavy = "26b" in model
        async with manager.acquire(model):
            async with lock:
                active += 1
                active_heavy += int(heavy)
                peak = max(peak, active)
                peak_heavy = max(peak_heavy, active_heavy)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
                active_heavy -= int(heavy)

    await asyncio.gather(
        work("gemma4:26b"),
        work("gemma4:26b"),
        work("qwen:4b"),
        work("qwen:4b"),
    )

    assert peak <= 2
    assert peak_heavy <= 1
    snapshot = await manager.snapshot()
    assert snapshot.active == snapshot.active_heavy == snapshot.queued == 0
