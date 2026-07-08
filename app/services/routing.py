from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.settings import Settings


URL_RE = re.compile(r"https?://[^\s)>'\"]+", re.IGNORECASE)
TICKER_RE = re.compile(r"(?:\$|\b)([A-Z]{1,5})(?:\b)")
ROAD_RE = re.compile(r"\b(?:I|US|SR|IN|KY|OH|MI|IL)[-\s]?\d{1,4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    tools: list[str] = field(default_factory=list)
    model_key: str = "general"
    needs_query_rewrite: bool = False
    reason: str = ""

    @property
    def uses_tools(self) -> bool:
        return bool(self.tools)


class ModelRouter:
    """Deterministic first-pass router for model and MCP tool selection.

    This is intentionally explicit and testable. The LLM is not trusted to
    decide when to send mail, call tools, or choose expensive models. Metadata
    can override tools for controlled tests/integrations, but unsafe operations
    still require explicit arguments in graph tool argument construction.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan(self, message: str, metadata: dict[str, Any] | None = None) -> QueryPlan:
        text = message.strip()
        lower = text.lower()
        metadata = metadata or {}

        forced_tools = _normalize_tool_list(metadata.get("force_tools") or metadata.get("tools"))
        if forced_tools:
            return QueryPlan(
                intent=str(metadata.get("intent") or "forced_tools"),
                tools=[tool for tool in forced_tools if tool in TOOLS_SUPPORTED],
                model_key=str(metadata.get("model_key") or "general"),
                needs_query_rewrite=bool(metadata.get("rewrite_query", False)),
                reason="Tools were explicitly requested by request metadata.",
            )

        urls = extract_urls(text)

        if _contains_any(lower, ["mcp health", "tool health", "health check", "check tools"]):
            return QueryPlan("tool_health", ["health_check"], "simple", False, "MCP health/tool check request.")

        if urls:
            if _contains_any(lower, ["image", "images", "photo", "picture", "thumbnail", "extract image"]):
                return QueryPlan(
                    intent="image_url_extraction",
                    tools=["extract_image_urls"],
                    model_key="vision",
                    needs_query_rewrite=False,
                    reason="URL plus image extraction intent.",
                )
            return QueryPlan(
                intent="url_scrape",
                tools=["scrape_url"],
                model_key="search",
                needs_query_rewrite=False,
                reason="URL provided; scrape before answering.",
            )

        if _contains_any(lower, ["weather", "forecast", "temperature", "rain", "snow", "storm", "wind", "humid"]):
            return QueryPlan("weather", ["weather_lookup"], "search", True, "Weather/current forecast request.")

        if _contains_any(lower, ["image", "screenshot", "photo", "picture", "chart", "diagram", "visual"]):
            return QueryPlan("vision", [], "vision", False, "Visual/image-related request; route to the vision model when image input is available.")

        if _contains_any(lower, ["embedding", "vector", "semantic search", "rag index", "similarity search"]):
            return QueryPlan("embedding_or_rag", [], "general", False, "Embedding/RAG planning request; embedding model is cataloged for vector tasks, chat answer uses a generative model.")

        if _contains_any(lower, ["road condition", "traffic", "closure", "closed road", "accident on", "construction on"]):
            return QueryPlan(
                "road_condition",
                ["road_condition_search"],
                "search",
                True,
                "Road condition requests require fresh official/state data.",
            )

        if _contains_any(lower, ["world cup", "match result", "match results", "score", "scores", "fixture", "fixtures", "standings", "who won", "final score"]):
            return QueryPlan(
                "sports_or_match_results",
                ["web_search_and_scrape", "news_search"],
                "search",
                True,
                "Sports/match results are current information and need search plus page evidence.",
            )

        if self._looks_like_stock_request(text):
            tools = ["stock_quote"]
            if _contains_any(lower, ["why", "move", "moved", "down", "up", "drop", "gain", "reason"]):
                tools = ["explain_stock_move"]
            elif _contains_any(lower, ["news", "latest", "recent", "today", "this week"]):
                tools = ["stock_quote", "stock_news"]
            return QueryPlan(
                "stock",
                tools,
                "search",
                True,
                "Stock/market request requires quote/news evidence.",
            )

        if _contains_any(lower, ["latest", "today", "current", "recent", "breaking", "news", "update", "this week"]):
            return QueryPlan("news", ["news_search"], "search", True, "Fresh news/current-events request.")

        if _contains_any(lower, ["quick search", "search only", "web search", "find links", "search result"]):
            return QueryPlan(
                "web_search",
                ["web_search"],
                "search",
                True,
                "Lightweight search request without page scraping.",
            )

        if _contains_any(lower, ["search", "internet", "web", "look up", "find online", "source", "reference"]):
            return QueryPlan(
                "web_research",
                ["web_search_and_scrape"],
                "search",
                True,
                "Explicit web research request.",
            )

        if _contains_any(lower, ["email", "mail", "inbox", "draft"]):
            if _contains_any(lower, ["send draft", "send the draft"]):
                return QueryPlan(
                    "mail_send_draft",
                    ["mail_send_draft"],
                    "general",
                    False,
                    "Draft-send request; graph requires draft_id and confirmation_token metadata.",
                )
            if _contains_any(lower, ["create draft", "draft an email", "email draft", "write draft"]):
                return QueryPlan(
                    "mail_create_draft",
                    ["mail_create_draft"],
                    "general",
                    False,
                    "Email draft request; send is never automatic.",
                )
            if _contains_any(lower, ["read", "open", "message id"]):
                return QueryPlan("mail_read", ["mail_read"], "general", False, "Mail read intent.")
            return QueryPlan("mail_search", ["mail_search"], "general", True, "Mail search intent.")

        if _contains_any(lower, ["classify", "classification", "intent", "routing", "query type"]):
            return QueryPlan(
                "classification",
                [],
                "classifier",
                False,
                "Classification/routing request; use the compact classifier model.",
            )

        if _contains_any(lower, ["rewrite", "polish", "draft", "write an email", "documentation", "readme", "report"]):
            return QueryPlan(
                "writing",
                [],
                "writer",
                False,
                "Writing/rewrite/documentation request; use the writer model.",
            )

        if _contains_any(lower, ["math", "algorithm", "logic", "prove", "calculate", "equation"]):
            return QueryPlan(
                "fast_reasoning",
                [],
                "fast_reasoning",
                False,
                "Math/logic/algorithm request; use the fast reasoning model first.",
            )

        if _contains_any(lower, ["architect", "architecture", "design", "debug", "explain why", "reason", "tradeoff", "compare", "analyze"]):
            return QueryPlan(
                "reasoning",
                [],
                "reasoning",
                False,
                "Moderate reasoning request without required fresh data.",
            )

        if len(text) < 120 and "?" not in text:
            return QueryPlan("simple", [], "simple", False, "Short simple request.")

        return QueryPlan("general", [], "general", False, "General assistant request.")

    def _looks_like_stock_request(self, text: str) -> bool:
        lower = text.lower()
        stock_words = ["stock", "share", "ticker", "market cap", "earnings", "price target", "nasdaq", "nyse"]
        if any(word in lower for word in stock_words):
            return True
        # Avoid treating all uppercase acronyms as stocks unless finance verbs are present.
        if _contains_any(lower, ["quote", "trading", "premarket", "after hours"]):
            return bool(extract_ticker(text))
        return False


TOOLS_SUPPORTED = {
    "health_check",
    "web_search",
    "web_search_and_scrape",
    "scrape_url",
    "extract_image_urls",
    "weather_lookup",
    "stock_quote",
    "stock_news",
    "explain_stock_move",
    "news_search",
    "road_condition_search",
    "mail_search",
    "mail_read",
    "mail_create_draft",
    "mail_send_draft",
}


def extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".,;]") for match in URL_RE.finditer(text)]


def extract_ticker(text: str) -> str | None:
    common_non_tickers = {"I", "A", "THE", "AND", "OR", "AI", "LLM", "USA", "API", "MCP", "AWS", "IN"}
    for match in TICKER_RE.finditer(text):
        token = match.group(1).upper()
        if token not in common_non_tickers:
            return token
    return None


def extract_road(text: str) -> str | None:
    match = ROAD_RE.search(text)
    return match.group(0).upper().replace(" ", "-") if match else None


def extract_location(text: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ["location", "city", "default_location"]:
        if metadata.get(key):
            return str(metadata[key])

    lower = text.lower()
    for marker in [" in ", " near ", " around ", " for "]:
        if marker in lower:
            idx = lower.rfind(marker)
            candidate = text[idx + len(marker) :].strip(" ?.!")
            if candidate:
                candidate = re.split(r"\b(today|tomorrow|this week|now|tonight|forecast|weather)\b", candidate, flags=re.I)[0]
                candidate = candidate.strip(" ,;:-")
                if candidate:
                    return candidate
    return "Indianapolis, IN"


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_tool_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
