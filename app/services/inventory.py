from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from app.llm.model_registry import capabilities_for
from app.logging_config import log_kv
from app.observability.metrics import metrics
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeInventory:
    models: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    cached: bool = False
    age_seconds: float = 0.0

    @property
    def model_names(self) -> list[str]:
        names: list[str] = []
        for item in self.models:
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for item in self.tools:
            name = item.get("name")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names

    def has_tool(self, name: str) -> bool:
        return name in self.tool_names

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": _copy_dicts(self.models),
            "model_names": self.model_names,
            "tools": _copy_dicts(self.tools),
            "tool_names": self.tool_names,
            "errors": dict(self.errors),
            "cached": self.cached,
            "age_seconds": round(self.age_seconds, 3),
        }


@dataclass
class _SourceCache:
    value: list[dict[str, Any]] | None = None
    cached_at: float = 0.0


class InventoryService:
    """Single-flight cache for live Ollama models and MCP tool definitions.

    Model and tool sources have independent stale caches. If MCP is temporarily
    unavailable while Ollama refresh succeeds, callers receive the new model list
    plus stale tools and an explicit MCP error rather than losing both sources.
    """

    def __init__(self, settings: Settings, ollama_client: Any, mcp_client: Any) -> None:
        self.settings = settings
        self.ollama = ollama_client
        self.mcp = mcp_client
        self._models = _SourceCache()
        self._tools = _SourceCache()
        self._lock = asyncio.Lock()

    async def load(self, *, force_refresh: bool = False) -> RuntimeInventory:
        now = monotonic()
        if not force_refresh and self._both_fresh(now):
            metrics.inc("inventory.cache_hit")
            return self._snapshot(cached=True, now=now)

        async with self._lock:
            now = monotonic()
            if not force_refresh and self._both_fresh(now):
                metrics.inc("inventory.cache_hit")
                return self._snapshot(cached=True, now=now)

            started = monotonic()
            model_task = self._load_models(force_refresh=force_refresh)
            tool_task = self._load_tools(force_refresh=force_refresh)
            models_result, tools_result = await asyncio.gather(model_task, tool_task)
            models, model_error, models_stale = models_result
            tools, tool_error, tools_stale = tools_result
            errors = {
                key: value
                for key, value in (
                    ("ollama", model_error),
                    ("mcp", tool_error),
                )
                if value
            }
            metrics.observe("inventory.refresh_seconds", monotonic() - started)
            metrics.inc("inventory.refresh")
            if errors:
                metrics.inc("inventory.refresh_with_errors")

            return RuntimeInventory(
                models=models,
                tools=tools,
                errors=errors,
                cached=models_stale or tools_stale,
                age_seconds=max(
                    self._source_age(self._models, monotonic()),
                    self._source_age(self._tools, monotonic()),
                ),
            )

    def invalidate(self) -> None:
        self._models = _SourceCache()
        self._tools = _SourceCache()
        invalidate_models = getattr(self.ollama, "invalidate_model_cache", None)
        if callable(invalidate_models):
            invalidate_models()
        invalidate_tools = getattr(self.mcp, "invalidate_tools_cache", None)
        if callable(invalidate_tools):
            invalidate_tools()

    def _both_fresh(self, now: float) -> bool:
        return self._source_fresh(self._models, now) and self._source_fresh(
            self._tools, now
        )

    def _source_fresh(self, source: _SourceCache, now: float) -> bool:
        return bool(
            source.value is not None
            and now - source.cached_at <= self.settings.inventory_cache_ttl_seconds
        )

    def _source_stale_usable(self, source: _SourceCache, now: float) -> bool:
        return bool(
            source.value is not None
            and now - source.cached_at
            <= self.settings.inventory_stale_if_error_seconds
        )

    async def _load_models(
        self,
        *,
        force_refresh: bool,
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        if self.settings.llm_backend != "ollama":
            self._models = _SourceCache(value=[], cached_at=monotonic())
            return [], None, False
        try:
            values = await self.ollama.list_models(
                force_refresh=force_refresh,
                allow_stale=False,
            )
            models = [dict(item) for item in values if isinstance(item, dict)]
            self._models = _SourceCache(value=models, cached_at=monotonic())
            return _copy_dicts(models), None, False
        except Exception as exc:  # dependency error is represented in inventory
            error = _format_exception(exc)
            log_kv(logger, logging.WARNING, "inventory_ollama_error", error=error)
            now = monotonic()
            if self._source_stale_usable(self._models, now):
                metrics.inc("inventory.ollama_stale_used")
                return _copy_dicts(self._models.value or []), error, True
            return [], error, False

    async def _load_tools(
        self,
        *,
        force_refresh: bool,
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        if not self.settings.mcp_enabled:
            self._tools = _SourceCache(value=[], cached_at=monotonic())
            return [], None, False
        try:
            values = await self.mcp.list_tools(
                force_refresh=force_refresh,
                allow_stale=False,
            )
            tools = [dict(item) for item in values if isinstance(item, dict)]
            self._tools = _SourceCache(value=tools, cached_at=monotonic())
            return _copy_dicts(tools), None, False
        except Exception as exc:  # dependency error is represented in inventory
            error = _format_exception(exc)
            log_kv(logger, logging.WARNING, "inventory_mcp_error", error=error)
            now = monotonic()
            if self._source_stale_usable(self._tools, now):
                metrics.inc("inventory.mcp_stale_used")
                return _copy_dicts(self._tools.value or []), error, True
            return [], error, False

    def _snapshot(self, *, cached: bool, now: float) -> RuntimeInventory:
        return RuntimeInventory(
            models=_copy_dicts(self._models.value or []),
            tools=_copy_dicts(self._tools.value or []),
            cached=cached,
            age_seconds=max(
                self._source_age(self._models, now),
                self._source_age(self._tools, now),
            ),
        )

    @staticmethod
    def _source_age(source: _SourceCache, now: float) -> float:
        if source.value is None or source.cached_at <= 0:
            return 0.0
        return max(0.0, now - source.cached_at)


class ModelSelector:
    """Resolve roles only to compatible live models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(
        self,
        key: str | None,
        inventory: RuntimeInventory | None = None,
    ) -> str:
        role = key or "general"
        configured = self.settings.model_for_key(role)
        available = inventory.model_names if inventory else []
        if not available:
            return configured
        for candidate in self._preferred_candidates(role):
            if candidate in available and self._compatible(role, candidate):
                return candidate
        fallback = self.settings.model_fallback
        if fallback in available and self._compatible(role, fallback):
            return fallback
        raise RuntimeError(
            f"No compatible installed Ollama model is available for role {role!r}."
        )

    def configured_roles(
        self,
        inventory: RuntimeInventory | None = None,
    ) -> dict[str, dict[str, Any]]:
        roles = [
            "planner",
            "simple",
            "general",
            "search",
            "reasoning",
            "fast_reasoning",
            "heavy",
            "synthesis",
            "writer",
            "classifier",
            "vision",
            "fallback",
        ]
        output: dict[str, dict[str, Any]] = {}
        for role in roles:
            configured = self.settings.model_for_key(role)
            try:
                resolved = self.resolve(role, inventory)
                error = None
            except RuntimeError as exc:
                resolved = None
                error = str(exc)
            output[role] = {
                "configured": configured,
                "resolved": resolved,
                "available": resolved is not None,
                "error": error,
            }

        available = set(inventory.model_names if inventory else [])
        output["embedding"] = {
            "configured": self.settings.embedding_model,
            "resolved": (
                self.settings.embedding_model
                if not available or self.settings.embedding_model in available
                else None
            ),
            "available": not available or self.settings.embedding_model in available,
            "chat_model": False,
        }
        return output

    def _compatible(self, role: str, model: str) -> bool:
        caps = capabilities_for(model)
        if role == "vision":
            return caps.vision
        if role == "embedding":
            return caps.embedding
        return caps.chat and not caps.embedding

    def _preferred_candidates(self, key: str) -> list[str]:
        preferences = {
            "planner": [
                self.settings.model_planner,
                self.settings.model_general,
                self.settings.model_classifier,
                self.settings.model_search,
            ],
            "simple": [
                self.settings.model_simple,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "general": [
                self.settings.model_general,
                self.settings.model_simple,
                self.settings.model_search,
                self.settings.model_fallback,
            ],
            "search": [
                self.settings.model_search,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "reasoning": [
                self.settings.model_reasoning,
                self.settings.model_fast_reasoning,
                self.settings.model_heavy,
                self.settings.model_search,
            ],
            "fast_reasoning": [
                self.settings.model_fast_reasoning,
                self.settings.model_reasoning,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "heavy": [
                self.settings.model_heavy,
                self.settings.model_synthesis,
                self.settings.model_reasoning,
                self.settings.model_search,
            ],
            "synthesis": [
                self.settings.model_synthesis,
                self.settings.model_heavy,
                self.settings.model_reasoning,
                self.settings.model_search,
            ],
            "writer": [
                self.settings.model_writer,
                self.settings.model_general,
                self.settings.model_synthesis,
                self.settings.model_fallback,
            ],
            "classifier": [
                self.settings.model_classifier,
                self.settings.model_general,
                self.settings.model_simple,
                self.settings.model_fallback,
            ],
            "vision": [self.settings.model_vision],
            "fallback": [
                self.settings.model_fallback,
                self.settings.model_general,
            ],
        }
        merged: list[str] = []
        for candidate in preferences.get(
            key,
            [self.settings.model_for_key(key), self.settings.model_fallback],
        ):
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged


def normalize_inventory(value: RuntimeInventory | dict[str, Any]) -> RuntimeInventory:
    """Normalize legacy graph inventory dictionaries for a safe transition."""

    if isinstance(value, RuntimeInventory):
        return value
    return RuntimeInventory(
        models=[item for item in value.get("models", []) if isinstance(item, dict)],
        tools=[item for item in value.get("tools", []) if isinstance(item, dict)],
        errors={str(k): str(v) for k, v in value.get("errors", {}).items()},
        cached=bool(value.get("cached", False)),
        age_seconds=float(value.get("age_seconds", 0.0) or 0.0),
    )


def build_inventory_payload(
    settings: Settings,
    inventory: RuntimeInventory | dict[str, Any],
    selector: ModelSelector | Any,
) -> dict[str, Any]:
    runtime = normalize_inventory(inventory)
    effective_selector = (
        selector if hasattr(selector, "configured_roles") else ModelSelector(settings)
    )
    return {
        "service": {
            "name": settings.app_name,
            "environment": settings.environment,
            "backend": settings.llm_backend,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "models": runtime.models,
            "model_names": runtime.model_names,
            "configured_roles": effective_selector.configured_roles(runtime),
            "model_task_catalog": settings.model_role_catalog(),
        },
        "mcp": {
            "enabled": settings.mcp_enabled,
            "server_url": settings.mcp_server_url,
            "tools": runtime.tools,
            "tool_names": runtime.tool_names,
        },
        "cache": {
            "cached": runtime.cached,
            "age_seconds": round(runtime.age_seconds, 3),
        },
        "errors": runtime.errors,
    }


def _copy_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
