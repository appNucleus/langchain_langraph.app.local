from __future__ import annotations

from app.services.query_rewriter import QueryRewriter
from app.services.routing import ModelRouter, extract_location, extract_ticker
from app.settings import Settings


def test_router_weather() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("Will it rain in Carmel tomorrow?")
    assert plan.intent == "weather"
    assert plan.tools == ["weather_lookup"]
    assert plan.model_key == "search"


def test_router_url_scrape() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("Summarize https://example.com/article")
    assert plan.intent == "url_scrape"
    assert plan.tools == ["scrape_url"]


def test_router_current_news() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("What is the latest news about NVIDIA?")
    assert plan.intent == "news"
    assert plan.tools == ["news_search"]


def test_extract_ticker() -> None:
    assert extract_ticker("Why did TSLA stock move?") == "TSLA"


def test_extract_location_from_metadata() -> None:
    assert extract_location("weather tomorrow", {"location": "Fishers, IN"}) == "Fishers, IN"


def test_rewriter_adds_weather_context() -> None:
    settings = Settings(default_forecast_days=5)
    router = ModelRouter(settings)
    plan = router.plan("weather in Indianapolis")
    rewritten = QueryRewriter(settings).rewrite("weather in Indianapolis", plan)
    assert "Indianapolis" in rewritten
    assert "next 5 days" in rewritten


def test_router_light_web_search_uses_web_search_tool() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("quick search LangGraph FastAPI examples")
    assert plan.intent == "web_search"
    assert plan.tools == ["web_search"]
    assert plan.needs_query_rewrite is True


def test_router_health_check() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("check tools health")
    assert plan.intent == "tool_health"
    assert plan.tools == ["health_check"]


def test_router_mail_send_draft_is_guarded_intent() -> None:
    router = ModelRouter(Settings())
    plan = router.plan("send draft email now")
    assert plan.intent == "mail_send_draft"
    assert plan.tools == ["mail_send_draft"]
