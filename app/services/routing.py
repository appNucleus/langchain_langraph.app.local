from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from app.services.inventory import ModelSelector, RuntimeInventory
from app.settings import Settings

_CURRENT_TERMS = {
    "current",
    "currently",
    "latest",
    "live",
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "recent",
    "recently",
    "now",
    "this week",
    "last night",
    "breaking",
}
_SEARCH_TERMS = {
    "find",
    "search",
    "look up",
    "research",
    "source",
    "evidence",
    "news",
    "weather",
    "forecast",
    "score",
    "match",
    "game",
    "fifa",
    "world cup",
    "critic",
    "criticism",
    "reaction",
    "report",
}
_REASONING_TERMS = {
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "critic",
    "criticism",
    "tradeoff",
    "risk",
    "recommend",
    "recommendation",
    "why",
    "evaluate",
    "architecture",
    "diagnostic",
}
_WRITE_MARKERS = {
    "send",
    "create",
    "delete",
    "remove",
    "update",
    "modify",
    "write",
    "draft",
    "post",
    "publish",
    "book",
    "purchase",
    "order",
    "cancel",
    "forward",
    "reply",
}


@dataclass(frozen=True)
class TaskRoutingDecision:
    requires_external_evidence: bool
    worker_role: str
    verifier_role: str
    reason: str


@dataclass(frozen=True)
class ModelRoutingDecision:
    role: str
    model: str
    reason: str


@dataclass(frozen=True)
class ToolRoutingDecision:
    name: str
    arguments: dict[str, Any]
    score: int
    reason: str


class RuntimeRouter:
    """Deterministic runtime router over live Ollama and MCP inventory.

    The router deliberately does not let an LLM invent unavailable model or tool
    names. It classifies the task, resolves the role through ``ModelSelector``,
    and ranks only MCP tools reported by the live inventory.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models = ModelSelector(settings)

    @staticmethod
    def execution_context() -> dict[str, str]:
        now = datetime.now(timezone.utc)
        return {
            "execution_time_utc": now.isoformat(),
            "execution_date_utc": now.date().isoformat(),
            "relative_date_rule": (
                "Resolve relative dates such as today/yesterday from execution_time_utc; "
                "do not invent a calendar date when external evidence is required."
            ),
        }

    def classify_task(
        self,
        *,
        user_request: str,
        task: dict[str, Any] | None,
    ) -> TaskRoutingDecision:
        task = task or {}
        required = task.get("required_evidence") or []
        objective = str(task.get("objective") or "")
        text = self._normalize(f"{user_request} {objective} {' '.join(map(str, required))}")

        current = self._contains_any(text, _CURRENT_TERMS)
        search = self._contains_any(text, _SEARCH_TERMS)
        explicitly_requires_evidence = bool(required)
        requires_external = current or search or explicitly_requires_evidence
        reasoning = self._contains_any(text, _REASONING_TERMS)

        if requires_external:
            worker_role = "search"
            reason = "current/search/evidence-dependent task"
        elif reasoning:
            worker_role = "reasoning"
            reason = "analysis/reasoning task"
        elif len(text.split()) <= 18:
            worker_role = "simple"
            reason = "short stable task"
        else:
            worker_role = "general"
            reason = "general task"

        verifier_role = "reasoning" if reasoning or requires_external else "fast_reasoning"
        return TaskRoutingDecision(
            requires_external_evidence=requires_external,
            worker_role=worker_role,
            verifier_role=verifier_role,
            reason=reason,
        )

    def select_model(
        self,
        *,
        role: str,
        inventory: RuntimeInventory,
        reason: str,
    ) -> ModelRoutingDecision:
        model = self.models.resolve(role, inventory)
        return ModelRoutingDecision(role=role, model=model, reason=reason)

    def select_tools(
        self,
        *,
        user_request: str,
        task: dict[str, Any] | None,
        required_actions: Iterable[str] = (),
        inventory: RuntimeInventory,
        metadata: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> list[ToolRoutingDecision]:
        metadata = metadata or {}
        task = task or {}
        action_text = " ".join(str(item) for item in required_actions if item)
        objective = str(task.get("objective") or "")
        query = " ".join(part for part in (objective, action_text, user_request) if part).strip()
        normalized = self._normalize(query)

        decisions: list[ToolRoutingDecision] = []
        for tool in inventory.tools:
            name = str(tool.get("name") or "").strip()
            if not name or not self._read_only_candidate(name):
                continue
            description = str(tool.get("description") or "")
            score, reasons = self._tool_score(name, description, normalized)
            arguments = self._arguments_for_tool(tool, query, metadata)
            if arguments is None:
                continue
            if score <= 0:
                continue
            decisions.append(
                ToolRoutingDecision(
                    name=name,
                    arguments=arguments,
                    score=score,
                    reason=", ".join(reasons) or "generic read-only research tool",
                )
            )

        decisions.sort(key=lambda item: (-item.score, item.name))
        return decisions[: max(1, limit)]

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.lower().split())

    @staticmethod
    def _contains_any(text: str, terms: set[str]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _read_only_candidate(name: str) -> bool:
        normalized = name.lower().replace("-", "_")
        parts = set(filter(None, re.split(r"[^a-z0-9]+", normalized)))
        return not bool(parts.intersection(_WRITE_MARKERS))

    def _tool_score(self, name: str, description: str, query: str) -> tuple[int, list[str]]:
        haystack = self._normalize(f"{name} {description}")
        score = 0
        reasons: list[str] = []

        if "weather" in query or "forecast" in query:
            score += self._score_terms(haystack, ("weather", "forecast", "climate"), 12)
            if any(term in haystack for term in ("weather", "forecast")):
                reasons.append("weather capability")

        if any(term in query for term in ("fifa", "football", "soccer", "match", "game", "score")):
            score += self._score_terms(
                haystack,
                ("sports", "football", "soccer", "fifa", "match", "score", "news"),
                10,
            )
            if any(term in haystack for term in ("sports", "football", "soccer", "news")):
                reasons.append("sports/news capability")

        if self._contains_any(query, _CURRENT_TERMS) or "news" in query:
            score += self._score_terms(haystack, ("news", "search", "web", "scrape"), 8)
            if any(term in haystack for term in ("news", "search", "web")):
                reasons.append("fresh-information capability")

        if any(term in query for term in ("critic", "reaction", "report", "source", "evidence")):
            score += self._score_terms(haystack, ("news", "search", "web", "scrape", "article"), 7)
            if any(term in haystack for term in ("news", "search", "scrape")):
                reasons.append("source-discovery capability")

        # General safe research fallback. Combined search+scrape tools are useful
        # when no domain-specific MCP tool exists.
        score += self._score_terms(haystack, ("search", "web", "news", "scrape", "fetch"), 3)
        if name in {"web_search_and_scrape", "web_search", "news_search"}:
            score += 4
            reasons.append("known read-only research fallback")
        return score, reasons

    @staticmethod
    def _score_terms(haystack: str, terms: tuple[str, ...], weight: int) -> int:
        return sum(weight for term in terms if term in haystack)

    def _arguments_for_tool(
        self,
        tool: dict[str, Any],
        query: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        required = schema.get("required") or []
        if not isinstance(required, list):
            required = []

        pages = self._safe_int(metadata.get("pages"), default=3, minimum=1, maximum=10)
        arguments: dict[str, Any] = {}
        for prop, definition in properties.items():
            if not isinstance(prop, str):
                continue
            prop_lower = prop.lower()
            definition = definition if isinstance(definition, dict) else {}
            value: Any | None = None
            if prop_lower in {"query", "q", "search_query", "search", "text", "prompt"}:
                value = query
            elif prop_lower in {"pages", "limit", "count", "max_results", "num_results"}:
                value = pages
            elif prop_lower in {"prefer_official", "official_only"}:
                value = True
            elif prop_lower in {"location", "city", "place"}:
                value = self._extract_location(query) or query
            elif prop_lower in metadata:
                value = metadata[prop_lower]
            elif prop in metadata:
                value = metadata[prop]
            elif "default" in definition:
                continue
            elif prop in required:
                value = self._fallback_required_value(definition, query, pages)
                if value is None:
                    return None

            if value is not None:
                arguments[prop] = value

        # Some MCP search tools omit a detailed schema. Preserve compatibility
        # with the repository's established search-tool contract.
        if not properties:
            arguments = {"query": query, "pages": pages, "prefer_official": True}

        return arguments

    @staticmethod
    def _fallback_required_value(
        definition: dict[str, Any], query: str, pages: int
    ) -> Any | None:
        value_type = definition.get("type")
        if value_type == "string" or value_type is None:
            return query
        if value_type == "integer":
            return pages
        if value_type == "number":
            return float(pages)
        if value_type == "boolean":
            return True
        return None

    @staticmethod
    def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _extract_location(query: str) -> str | None:
        match = re.search(
            r"\b(?:in|for|at|near)\s+([A-Z][A-Za-z .'-]+?)(?:[?.!,]|\s+(?:today|tomorrow|yesterday|now)\b|$)",
            query,
        )
        return match.group(1).strip() if match else None


def build_fresh_run_state(
    *,
    message: str,
    system_prompt: str,
    metadata: dict[str, Any],
    history: list[dict[str, Any]],
    execution_budget: Any,
    request_id: str,
    inventory: RuntimeInventory,
    backend: str,
) -> dict[str, Any]:
    """Return a complete per-invocation state that overwrites stale checkpoints.

    LangGraph merges supplied input with the latest checkpoint for a thread. Every
    transient channel is therefore included explicitly, including ``None`` and
    empty containers, so a completed/terminated prior run cannot affect the new
    request while external conversation history remains available separately.
    """

    return {
        "message": message,
        "system_prompt": system_prompt,
        "metadata": dict(metadata),
        "history": list(history),
        "execution_budget": execution_budget,
        "request_id": request_id,
        "inventory": inventory.as_dict(),
        "routing": {},
        "selected_models": {},
        "selected_tool": None,
        "selected_tools": {},
        "researched_task_ids": [],
        "plan": {},
        "task_index": 0,
        "task_results": [],
        "worker_result": {},
        "verification": {},
        "evidence": [],
        "iterations": 0,
        "research_rounds": 0,
        "replans": 0,
        "next_action": "",
        "response": "",
        "backend": backend,
        "model": None,
        "termination_reason": None,
    }
