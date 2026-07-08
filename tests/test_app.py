from __future__ import annotations

from fastapi.testclient import TestClient

from app.factory import create_app
from app.graph import ChatAgent
from app.settings import Settings
from tests.conftest import FakeMCPClient, FakeOllamaClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "ollama"
    assert payload["models"]["search"] == "qwen3.5:9b"
    assert payload["mcp_follow_redirects"] is True


def test_chat_uses_general_model_for_general_request(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    response = client.post("/api/chat", json={"message": "Explain this architecture in a few sentences."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "ollama"
    assert payload["model"] == "phi4-mini-reasoning:latest"
    assert "Answer from" in payload["response"]
    assert fake_ollama.calls[-1]["model"] == "phi4-mini-reasoning:latest"


def test_weather_request_calls_weather_tool_and_search_model(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "What is the weather in Indianapolis tomorrow?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen3.5:9b"
    assert payload["metadata"]["intent"] == "weather"
    assert payload["metadata"]["tools_used"] == ["weather_lookup"]
    assert fake_mcp.calls[-1]["name"] == "weather_lookup"
    assert "Indianapolis" in fake_mcp.calls[-1]["arguments"]["location"]


def test_news_request_rewrites_query_before_tool_call(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "latest OpenAI news"},
    )
    assert response.status_code == 200
    call = fake_mcp.calls[-1]
    assert call["name"] == "news_search"
    assert "latest news" in call["arguments"]["query"].lower()
    assert "authoritative sources" in call["arguments"]["query"].lower()
    assert response.json()["metadata"]["references"][0]["url"] == "https://news.example/item"


def test_stock_move_routes_to_explain_stock_move(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Why did TSLA stock move today?"},
    )
    assert response.status_code == 200
    assert fake_mcp.calls[-1]["name"] == "explain_stock_move"
    assert fake_mcp.calls[-1]["arguments"]["symbol"] == "TSLA"


def test_stream_endpoint_returns_sse_events(client: TestClient) -> None:
    with client.stream("POST", "/api/chat/stream", json={"message": "hello"}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "event: status" in text
    assert "event: plan" in text
    assert "event: token" in text
    assert "event: done" in text
    assert "stream " in text
    assert "answer" in text


def test_api_key_required() -> None:
    settings = Settings(llm_backend="echo", api_key="secret", mcp_enabled=False)
    app = create_app(settings=settings, chat_agent=ChatAgent(settings))
    client = TestClient(app)

    missing = client.post("/api/chat", json={"message": "hello"})
    assert missing.status_code == 401

    ok = client.post("/api/chat", json={"message": "hello"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_chat_validation_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_live_health_with_fake_dependencies(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ollama"]["root"] == "Ollama is running"
    assert payload["mcp"]["ok"] is True


def test_light_search_uses_web_search_tool(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post("/api/chat", json={"message": "quick search LangGraph FastAPI"})
    assert response.status_code == 200
    assert fake_mcp.calls[-1]["name"] == "web_search"
    assert "current authoritative sources" in fake_mcp.calls[-1]["arguments"]["query"]


def test_mail_send_draft_requires_confirmation_metadata(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "send draft email now"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["intent"] == "mail_send_draft"
    assert payload["metadata"]["tools_used"] == []
    assert payload["metadata"]["tool_errors"][0]["tool"] == "mail_send_draft"


def test_forced_tools_can_cover_supported_mcp_surface(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={
            "message": "Run controlled tool coverage test.",
            "metadata": {
                "force_tools": ["health_check", "web_search", "mail_read"],
                "rewrite_query": True,
                "message_id": "msg-123",
            },
        },
    )
    assert response.status_code == 200
    names = [call["name"] for call in fake_mcp.calls[-3:]]
    assert names == ["health_check", "web_search", "mail_read"]


def test_world_cup_match_results_uses_search_and_scrape(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "What is todays world cup match results in detail"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["intent"] == "sports_or_match_results"
    assert payload["metadata"]["tools_requested"] == ["web_search_and_scrape", "news_search"]
    names = [call["name"] for call in fake_mcp.calls[-2:]]
    assert names == ["web_search_and_scrape", "news_search"]
    rewritten = payload["metadata"]["rewritten_query"].lower()
    assert "final score" in rewritten
    assert "official match centre" in rewritten


def test_echo_backend_skips_real_mcp_calls(fake_mcp: FakeMCPClient) -> None:
    settings = Settings(llm_backend="echo", mcp_enabled=True, api_key="")
    app = create_app(settings=settings, chat_agent=ChatAgent(settings, mcp_client=fake_mcp))
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "latest OpenAI news"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "echo"
    assert payload["metadata"]["tools_requested"] == ["news_search"]
    assert payload["metadata"]["tools_used"] == []
    assert fake_mcp.calls == []
