from __future__ import annotations

import os

import pytest

from app.llm.ollama import OllamaClient
from app.mcp.client import MCPClient
from app.settings import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("LIVE_INTEGRATION") != "1",
    reason="Set LIVE_INTEGRATION=1 to run tests against live local Ollama/MCP services.",
)


@pytest.mark.asyncio
async def test_live_ollama_tags_and_chat() -> None:
    settings = Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama.home.arpa:11434"),
        llm_backend="ollama",
    )
    client = OllamaClient(settings)
    health = await client.health()
    assert "Ollama" in health["root"]
    assert settings.model_general in health["models"]

    result = await client.chat(
        model=settings.model_general,
        messages=[
            {"role": "system", "content": "Reply with a very short final answer. Do not include reasoning."},
            {"role": "user", "content": "Say OK."},
        ],
        temperature=0,
        num_predict=128,
    )
    assert result.model == settings.model_general
    assert result.content.strip()
    assert "thinking" not in result.content.lower()


@pytest.mark.asyncio
async def test_live_ollama_streaming_content() -> None:
    settings = Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama.home.arpa:11434"),
        llm_backend="ollama",
        ollama_stream_num_predict=128,
    )
    client = OllamaClient(settings)
    chunks = [
        chunk
        async for chunk in client.stream_chat(
            model=settings.model_general,
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
            temperature=0,
            num_predict=128,
        )
    ]
    assert "".join(chunks).strip()


@pytest.mark.asyncio
async def test_live_mcp_list_tools() -> None:
    settings = Settings(
        mcp_server_url=os.getenv("MCP_SERVER_URL", "https://mcp.home.arpa/mcp"),
        mcp_verify_tls=os.getenv("MCP_VERIFY_TLS", "false").lower() == "true",
    )
    client = MCPClient(settings)
    tools = await client.list_tools()
    tool_names = {tool.get("name") for tool in tools}
    assert "health_check" in tool_names
    assert "web_search" in tool_names
    assert "weather_lookup" in tool_names
