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


def test_chat_uses_reasoning_model_for_architecture_request(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    response = client.post("/api/chat", json={"message": "Explain this architecture in a few sentences."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "ollama"
    assert payload["model"] == "deepseek-r1:8b"
    assert payload["metadata"]["intent"] == "reasoning"
    assert "Answer from" in payload["response"]
    assert fake_ollama.calls[-1]["model"] == "deepseek-r1:8b"


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
    assert "Answer from" in text
    assert "event: inventory" not in text


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


def test_inventory_endpoint_lists_live_models_tools_and_full_role_catalog(client: TestClient) -> None:
    response = client.get("/api/inventory")
    assert response.status_code == 200
    payload = response.json()
    assert "qwen3.5:4b" in payload["ollama"]["model_names"]
    assert payload["ollama"]["configured_roles"]["general"]["resolved"] == "qwen3.5:4b"
    assert payload["ollama"]["configured_roles"]["simple"]["resolved"] == "qwen3.5:2b"
    assert payload["ollama"]["configured_roles"]["search"]["resolved"] == "qwen3.5:9b"
    assert payload["ollama"]["configured_roles"]["reasoning"]["resolved"] == "deepseek-r1:8b"
    assert payload["ollama"]["configured_roles"]["fast_reasoning"]["resolved"] == "phi4-mini-reasoning:latest"
    assert payload["ollama"]["configured_roles"]["synthesis"]["resolved"] == "gemma4:12b-it-qat"
    assert payload["ollama"]["configured_roles"]["heavy"]["resolved"] == "gemma4:26b-a4b-it-qat"
    assert payload["ollama"]["configured_roles"]["writer"]["resolved"] == "gemma4:e4b-it-qat"
    assert payload["ollama"]["configured_roles"]["classifier"]["resolved"] == "gemma4:e2b-it-qat"
    assert payload["ollama"]["configured_roles"]["fallback"]["resolved"] == "granite3.3:8b"
    assert payload["ollama"]["configured_roles"]["embedding"]["resolved"] == "qwen3-embedding:0.6b"
    catalog_models = {item["model"] for item in payload["ollama"]["model_task_catalog"]}
    assert set(payload["ollama"]["model_names"]).issubset(catalog_models)
    assert "web_search_and_scrape" in payload["mcp"]["tool_names"]
    assert payload["errors"] == {}


def test_compound_request_is_split_answered_and_synthesized(client: TestClient, fake_mcp: FakeMCPClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "What is the latest OpenAI news? Also what is the weather in Indianapolis tomorrow?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["metadata"]["subqueries"]) >= 2
    assert payload["metadata"]["subqueries"][0]["tools"]
    assert payload["metadata"]["task_answers"]
    called_tools = [call["name"] for call in fake_mcp.calls]
    assert "news_search" in called_tools
    assert "weather_lookup" in called_tools


def test_chat_metadata_is_compact_without_full_inventory(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert "inventory" not in metadata
    assert "inventory_summary" in metadata
    assert "subqueries" in metadata
    assert "task_answers" in metadata


def test_validation_retries_broken_subquery_answer(settings: Settings, fake_mcp: FakeMCPClient) -> None:
    class BrokenThenGoodOllama(FakeOllamaClient):
        async def chat(self, *, model, messages, temperature=None, num_predict=None):  # type: ignore[no-untyped-def]
            self.calls.append({"model": model, "messages": list(messages), "temperature": temperature, "num_predict": num_predict})
            if len(self.calls) == 1:
                return type("Resp", (), {"content": "**", "model": model, "raw": {}})()
            return type("Resp", (), {
                "content": "This retry answer is complete and useful enough to pass validation after the first broken model output.",
                "model": model,
                "raw": {},
            })()

    local_settings = settings.model_copy(update={"enable_llm_query_planning": False})
    ollama = BrokenThenGoodOllama()
    agent = ChatAgent(local_settings, ollama_client=ollama, mcp_client=fake_mcp)
    local_client = TestClient(create_app(settings=local_settings, chat_agent=agent))
    response = local_client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["task_answers"][0]["retry_count"] == 1
    assert metadata["task_answers"][0]["validation"]["ok"] is True
    assert len(ollama.calls) >= 2
