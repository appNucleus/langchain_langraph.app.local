from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.routing import QueryPlan, extract_location, extract_road, extract_ticker
from app.settings import Settings


class QueryRewriter:
    """Deterministic query optimizer for MCP search tools.

    This deliberately does not call the LLM before search. It is fast,
    predictable, testable, and avoids spending tokens just to create a search
    phrase. The final answering LLM still receives both the original user
    question and the optimized query.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rewrite(self, message: str, plan: QueryPlan, *, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        base = self._clean(message)
        today = datetime.now(ZoneInfo("America/Indiana/Indianapolis")).strftime("%Y-%m-%d")

        if plan.intent == "weather":
            location = extract_location(message, metadata)
            return f"{location} weather forecast current conditions next {self.settings.default_forecast_days} days {today}"

        if plan.intent == "stock":
            ticker = extract_ticker(message) or base
            return f"{ticker} stock quote latest news earnings analyst guidance price movement last {self.settings.default_news_lookback_days} days {today}"

        if plan.intent == "sports_or_match_results":
            return (
                f"{base} today match results final score highlights schedule official match centre "
                f"ESPN FIFA ICC reliable sources {today}"
            )

        if plan.intent == "news":
            return f"{base} latest news recent developments authoritative sources published within last {self.settings.default_news_lookback_days} days {today}"

        if plan.intent == "road_condition":
            road = extract_road(message) or "road"
            location = extract_location(message, metadata)
            return f"official road conditions {road} {location} closures traffic construction {today}"

        if plan.intent == "mail_search":
            return base

        if plan.intent == "web_search":
            return f"{base} concise search results current authoritative sources"

        if plan.intent == "web_research":
            return f"{base} authoritative sources official documentation recent reliable references"

        return base

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip("?.! ")
        if len(text) > 350:
            return text[:350].rsplit(" ", 1)[0]
        return text
