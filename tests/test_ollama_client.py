from __future__ import annotations

import json

import httpx
import pytest

from app.llm.ollama import OllamaClient
from app.settings import Settings


@pytest.mark.asyncio
async def test_ollama_chat_uses_think_false_and_returns_content_only() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["think"] is False
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "message": {"role": "assistant", "content": "final answer", "thinking": "do not expose"},
                "done": True,
            },
        )

    client = OllamaClient(Settings(ollama_base_url="http://ollama.test:11434"), transport=httpx.MockTransport(handler))
    result = await client.chat(model="qwen3.5:4b", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "final answer"


@pytest.mark.asyncio
async def test_ollama_stream_skips_thinking_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "", "thinking": "hidden"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "hello"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": " world"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        return httpx.Response(200, text="\n".join(lines))

    client = OllamaClient(Settings(ollama_base_url="http://ollama.test:11434"), transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in client.stream_chat(model="qwen3.5:4b", messages=[{"role": "user", "content": "hi"}])]
    assert chunks == ["hello", " world"]


@pytest.mark.asyncio
async def test_ollama_list_models_reads_api_tags() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:4b"}]})

    client = OllamaClient(Settings(ollama_base_url="http://ollama.test:11434"), transport=httpx.MockTransport(handler))
    models = await client.list_models()
    assert models == [{"name": "qwen3.5:4b"}]
