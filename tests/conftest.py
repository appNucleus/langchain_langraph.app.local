from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.factory import create_app
from app.llm.ollama import LLMResponse
from app.mcp.client import MCPToolResult
from app.settings import Settings


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {"root": "Ollama is running", "models": ["qwen3.5:4b"]}

    async def chat(self, *, model: str, messages: Sequence[dict[str, str]], temperature: float | None = None, num_predict: int | None = None) -> LLMResponse:
        self.calls.append({"model": model, "messages": list(messages), "temperature": temperature, "num_predict": num_predict})
        return LLMResponse(
            content=f"Answer from {model}",
            model=model,
            raw={"message": {"content": f"Answer from {model}", "thinking": "hidden"}},
        )

    async def stream_chat(self, *, model: str, messages: Sequence[dict[str, str]], temperature: float | None = None, num_predict: int | None = None) -> AsyncIterator[str]:
        self.stream_calls.append({"model": model, "messages": list(messages), "temperature": temperature, "num_predict": num_predict})
        for chunk in ["stream ", "answer"]:
            yield chunk

    async def embed(self, *, model: str, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "weather_lookup"}]

    async def health_check(self) -> MCPToolResult:
        return MCPToolResult(tool="health_check", ok=True, data={"status": "ok"})

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        args = arguments or {}
        self.calls.append({"name": name, "arguments": args})
        if name == "weather_lookup":
            return MCPToolResult(
                tool=name,
                ok=True,
                data={
                    "location": args.get("location"),
                    "forecast": "Sunny",
                    "url": "https://weather.example/test",
                    "title": "Weather Example",
                },
            )
        if name == "news_search":
            return MCPToolResult(
                tool=name,
                ok=True,
                data={
                    "query": args.get("query"),
                    "results": [
                        {
                            "title": "Example News",
                            "url": "https://news.example/item",
                            "snippet": "Current event summary.",
                        }
                    ],
                },
            )
        if name == "web_search_and_scrape":
            return MCPToolResult(
                tool=name,
                ok=True,
                data={
                    "query": args.get("query"),
                    "results": [
                        {"title": "Web Result", "url": "https://web.example/result", "content": "Useful content"}
                    ],
                },
            )
        if name == "stock_quote":
            return MCPToolResult(tool=name, ok=True, data={"symbol": args.get("symbol"), "price": 123.45})
        if name == "stock_news":
            return MCPToolResult(tool=name, ok=True, data={"symbol": args.get("symbol"), "results": []})
        if name == "explain_stock_move":
            return MCPToolResult(tool=name, ok=True, data={"symbol": args.get("symbol"), "signals": ["news"]})
        return MCPToolResult(tool=name, ok=True, data={"args": args})


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        api_key="",
        ollama_base_url="http://ollama.test:11434",
        mcp_server_url="https://mcp.test/mcp",
        ollama_num_predict=256,
        ollama_stream_num_predict=256,
    )


@pytest.fixture()
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture()
def fake_mcp() -> FakeMCPClient:
    return FakeMCPClient()


@pytest.fixture()
def client(settings: Settings, fake_ollama: FakeOllamaClient, fake_mcp: FakeMCPClient) -> TestClient:
    from app.graph import ChatAgent

    agent = ChatAgent(settings, ollama_client=fake_ollama, mcp_client=fake_mcp)
    return TestClient(create_app(settings=settings, chat_agent=agent))
