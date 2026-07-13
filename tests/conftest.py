from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.factory import create_app
from app.llm.ollama import LLMResponse
from app.mcp.client import MCPToolResult
from app.settings import Settings

_TEST_CHAT_REQUEST_EXAMPLE: dict[str, Any] = {
    "message": "OpenAPI example injected by pytest",
    "thread_id": None,
    "conversation_id": None,
    "run_id": None,
    "resume": False,
    "resume_token": None,
    "system_prompt": None,
    "metadata": {},
}


@pytest.fixture()
def in_memory_chat_request_example() -> dict[str, Any]:
    """Return code-only OpenAPI test data; tests never read request JSON files."""

    return deepcopy(_TEST_CHAT_REQUEST_EXAMPLE)


@pytest.fixture(autouse=True)
def isolate_request_example_files(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_chat_request_example: dict[str, Any],
) -> None:
    """Keep every pytest app instance independent from runtime JSON documentation."""

    monkeypatch.setattr(
        "app.factory.load_chat_request_example",
        lambda: deepcopy(in_memory_chat_request_example),
    )


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {"root": "Ollama is running", "models": [item["name"] for item in await self.list_models()]}

    async def list_models(self) -> list[dict[str, Any]]:
        return [
            {"name": "gemma4:26b-a4b-it-qat", "size": 15000000000},
            {"name": "qwen3-vl:4b", "size": 3300000000},
            {"name": "phi4-mini-reasoning:latest", "size": 3200000000},
            {"name": "deepseek-r1:8b", "size": 5200000000},
            {"name": "qwen3.5:9b", "size": 6600000000},
            {"name": "qwen3.5:4b", "size": 3400000000},
            {"name": "qwen3.5:2b", "size": 2700000000},
            {"name": "granite3.3:8b", "size": 4900000000},
            {"name": "gemma4:12b-it-qat", "size": 7200000000},
            {"name": "gemma4:e4b-it-qat", "size": 6100000000},
            {"name": "gemma4:e2b-it-qat", "size": 4300000000},
            {"name": "qwen3-embedding:0.6b", "size": 639000000},
        ]

    async def chat(self, *, model: str, messages: Sequence[dict[str, str]], temperature: float | None = None, num_predict: int | None = None) -> LLMResponse:
        self.calls.append({"model": model, "messages": list(messages), "temperature": temperature, "num_predict": num_predict})
        content = f"Answer from {model}. This is a complete test answer with enough useful detail to pass validation for the requested query."
        return LLMResponse(
            content=content,
            model=model,
            raw={"message": {"content": content, "thinking": "hidden"}},
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
        return [
            {"name": "health_check", "description": "Check MCP health."},
            {"name": "web_search", "description": "Search the web."},
            {"name": "web_search_and_scrape", "description": "Search and scrape pages."},
            {"name": "scrape_url", "description": "Scrape a URL."},
            {"name": "extract_image_urls", "description": "Extract images from a URL."},
            {"name": "weather_lookup", "description": "Get current weather and forecast."},
            {"name": "stock_quote", "description": "Get stock quote."},
            {"name": "stock_news", "description": "Get stock news."},
            {"name": "explain_stock_move", "description": "Explain stock moves."},
            {"name": "news_search", "description": "Search recent news."},
            {"name": "road_condition_search", "description": "Search road conditions."},
            {"name": "mail_search", "description": "Search mail."},
            {"name": "mail_read", "description": "Read mail."},
            {"name": "mail_create_draft", "description": "Create mail draft."},
            {"name": "mail_send_draft", "description": "Send confirmed draft."},
        ]

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
