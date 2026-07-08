from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.llm.ollama import OllamaClient
from app.logging_config import log_kv
from app.services.inventory import ModelSelector, RuntimeInventory
from app.services.routing import ModelRouter, QueryPlan
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTask:
    id: str
    query: str
    plan: QueryPlan
    model: str
    tools: list[str] = field(default_factory=list)
    rewritten_query: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "intent": self.plan.intent,
            "model": self.model,
            "model_key": self.plan.model_key,
            "tools": self.tools,
            "rewritten_query": self.rewritten_query,
            "reason": self.plan.reason,
        }


class QueryDecomposer:
    """Turns one user request into a small list of answerable tasks.

    It first asks a fast/balanced planning model for a JSON plan. If that fails,
    it falls back to deterministic splitting, so unit tests and offline usage are
    stable.
    """

    def __init__(self, settings: Settings, router: ModelRouter, selector: ModelSelector) -> None:
        self.settings = settings
        self.router = router
        self.selector = selector

    async def decompose(
        self,
        *,
        message: str,
        metadata: dict[str, Any],
        inventory: RuntimeInventory,
        ollama: OllamaClient,
    ) -> list[str]:
        explicit = metadata.get("subqueries") or metadata.get("queries")
        if isinstance(explicit, list):
            cleaned = [_clean_query(str(item)) for item in explicit if _clean_query(str(item))]
            if cleaned:
                return cleaned[: self.settings.max_subqueries]

        if self.settings.llm_backend == "ollama" and self.settings.enable_llm_query_planning:
            planned = await self._llm_decompose(message=message, inventory=inventory, ollama=ollama)
            if planned:
                return planned[: self.settings.max_subqueries]

        return _heuristic_decompose(message, max_items=self.settings.max_subqueries)

    async def _llm_decompose(self, *, message: str, inventory: RuntimeInventory, ollama: OllamaClient) -> list[str]:
        model = self.selector.resolve("planner", inventory)
        prompt = (
            "Split the user's request into simple standalone questions only when it truly contains multiple tasks. "
            "Keep broad context, do not over-split small phrases, and return only valid JSON.\n"
            "JSON schema: {\"queries\": [\"simple question 1\", \"simple question 2\"]}\n"
            f"Available tools: {inventory.tool_names}\n"
            f"User request: {message}"
        )
        try:
            response = await ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a query planning service. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                num_predict=min(512, self.settings.ollama_num_predict),
            )
        except Exception as exc:  # noqa: BLE001
            log_kv(logger, logging.WARNING, "llm_decompose_error", model=model, error=str(exc))
            return []

        try:
            payload = _extract_json_object(response.content)
        except ValueError:
            log_kv(logger, logging.INFO, "llm_decompose_non_json", model=model)
            return []
        queries = payload.get("queries")
        if not isinstance(queries, list):
            return []
        cleaned = [_clean_query(str(item)) for item in queries if _clean_query(str(item))]
        if not cleaned:
            return []
        return cleaned

    def build_tasks(
        self,
        *,
        queries: list[str],
        metadata: dict[str, Any],
        inventory: RuntimeInventory,
    ) -> list[QueryTask]:
        tasks: list[QueryTask] = []
        for index, query in enumerate(queries, 1):
            plan = self.router.plan(query, metadata)
            selected_tools = select_available_tools(plan, query=query, inventory=inventory, settings=self.settings)
            plan = QueryPlan(
                intent=plan.intent,
                tools=selected_tools,
                model_key=plan.model_key,
                needs_query_rewrite=plan.needs_query_rewrite,
                reason=plan.reason,
            )
            model = self.selector.resolve(plan.model_key, inventory)
            tasks.append(QueryTask(id=f"q{index}", query=query, plan=plan, model=model, tools=selected_tools))
        return tasks


def select_available_tools(plan: QueryPlan, *, query: str, inventory: RuntimeInventory, settings: Settings) -> list[str]:
    available = set(inventory.tool_names)
    if not settings.mcp_enabled or not available:
        return []

    selected: list[str] = []
    for tool in plan.tools:
        _append_with_fallbacks(selected, tool, available)

    if not selected and settings.search_unknown_or_fresh and _looks_fresh_or_discovery_query(query):
        for candidate in ["web_search_and_scrape", "web_search", "news_search"]:
            if candidate in available:
                selected.append(candidate)
                break

    deduped: list[str] = []
    for tool in selected:
        if tool in available and tool not in deduped:
            deduped.append(tool)
    return deduped[: settings.max_tool_calls]


def _append_with_fallbacks(selected: list[str], tool: str, available: set[str]) -> None:
    fallback_map = {
        "web_search_and_scrape": ["web_search_and_scrape", "web_search", "news_search"],
        "news_search": ["news_search", "web_search_and_scrape", "web_search"],
        "weather_lookup": ["weather_lookup", "web_search_and_scrape", "web_search"],
        "road_condition_search": ["road_condition_search", "web_search_and_scrape", "web_search"],
        "stock_quote": ["stock_quote", "explain_stock_move", "web_search_and_scrape", "web_search"],
        "stock_news": ["stock_news", "news_search", "web_search_and_scrape", "web_search"],
        "explain_stock_move": ["explain_stock_move", "stock_news", "stock_quote", "web_search_and_scrape", "web_search"],
    }
    for candidate in fallback_map.get(tool, [tool]):
        if candidate in available and candidate not in selected:
            selected.append(candidate)
            return


def _heuristic_decompose(message: str, *, max_items: int) -> list[str]:
    cleaned = _clean_query(message)
    if not cleaned:
        return []

    # Hard separators usually mean independent tasks.
    pieces = _split_by_patterns(cleaned, [r"\n+", r"\s*;\s*", r"\s+\b(?:also|then)\b\s+"])

    # Multiple question marks usually indicate multiple questions.
    if len(pieces) == 1 and cleaned.count("?") >= 2:
        pieces = [part + "?" for part in cleaned.split("?") if part.strip()]

    # Split obvious compound asks, but avoid phrases like "pros and cons".
    if len(pieces) == 1 and _safe_to_split_on_and(cleaned):
        pieces = re.split(r"\s+and\s+(?=(?:what|why|how|when|where|who|can|do|does|is|are|should|please)\b)", cleaned, flags=re.I)

    output: list[str] = []
    for piece in pieces:
        item = _clean_query(piece)
        if item and item not in output:
            output.append(item)
        if len(output) >= max_items:
            break
    return output or [cleaned]


def _split_by_patterns(text: str, patterns: list[str]) -> list[str]:
    pieces = [text]
    for pattern in patterns:
        new_pieces: list[str] = []
        for piece in pieces:
            new_pieces.extend(re.split(pattern, piece, flags=re.I))
        pieces = [item.strip() for item in new_pieces if item.strip()]
    return pieces


def _safe_to_split_on_and(text: str) -> bool:
    lowered = text.lower()
    blocked = ["pros and cons", "bread and butter", "quality and speed", "fast and easy"]
    return " and " in lowered and not any(item in lowered for item in blocked)


def _looks_fresh_or_discovery_query(text: str) -> bool:
    lowered = text.lower()
    fresh_words = [
        "latest",
        "today",
        "current",
        "recent",
        "news",
        "price",
        "schedule",
        "version",
        "release",
        "available",
        "online",
        "search",
        "look up",
        "unknown",
        "not sure",
    ]
    discovery_starters = ("what is ", "who is ", "where is ", "when is ", "how much ", "which ")
    return any(word in lowered for word in fresh_words) or lowered.startswith(discovery_starters)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON payload is not an object")
    return data


def _clean_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-•")
    return text[:600].strip()
