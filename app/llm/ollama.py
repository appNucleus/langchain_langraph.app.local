from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any

import httpx2 as httpx

from app.llm.errors import OllamaError
from app.llm.resource_manager import OllamaResourceManager
from app.logging_config import log_kv
from app.observability.metrics import metrics
from app.orchestration.execution_meter import get_current_execution_meter
from app.settings import Settings

logger = logging.getLogger(__name__)

ChatMessage = dict[str, str]
StructuredFormat = str | Mapping[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class _RuntimeKey:
    base_url: str
    connect_timeout: float
    request_timeout: float
    pool_timeout: float
    max_connections: int
    max_keepalive_connections: int
    keepalive_expiry: float
    max_concurrency: int
    heavy_max_concurrency: int


class _SharedOllamaRuntime:
    """One HTTP pool, admission controller, and model cache per configuration."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resource_manager: OllamaResourceManager | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.resources = resource_manager or OllamaResourceManager(settings)
        self.client: httpx.AsyncClient | None = None
        self.client_lock = asyncio.Lock()
        self.models_lock = asyncio.Lock()
        self.models_cache: list[dict[str, Any]] | None = None
        self.models_cached_at = 0.0

    async def get_client(self) -> httpx.AsyncClient:
        if self.client is not None:
            return self.client
        async with self.client_lock:
            if self.client is None:
                timeout = httpx.Timeout(
                    connect=self.settings.ollama_connect_timeout_seconds,
                    read=self.settings.ollama_timeout_seconds,
                    write=self.settings.ollama_timeout_seconds,
                    pool=self.settings.ollama_pool_timeout_seconds,
                )
                limits = httpx.Limits(
                    max_connections=self.settings.ollama_max_connections,
                    max_keepalive_connections=(
                        self.settings.ollama_max_keepalive_connections
                    ),
                    keepalive_expiry=self.settings.http_keepalive_expiry_seconds,
                )
                self.client = httpx.AsyncClient(
                    base_url=self.settings.ollama_base_url.rstrip("/"),
                    timeout=timeout,
                    transport=self.transport,
                    limits=limits,
                )
        return self.client

    async def close(self) -> None:
        async with self.client_lock:
            client, self.client = self.client, None
        if client is not None:
            await client.aclose()

    def invalidate_models(self) -> None:
        self.models_cache = None
        self.models_cached_at = 0.0


_SHARED_RUNTIMES: dict[_RuntimeKey, _SharedOllamaRuntime] = {}
_SHARED_RUNTIMES_LOCK = RLock()


def _runtime_key(settings: Settings) -> _RuntimeKey:
    return _RuntimeKey(
        base_url=settings.ollama_base_url.rstrip("/"),
        connect_timeout=settings.ollama_connect_timeout_seconds,
        request_timeout=settings.ollama_timeout_seconds,
        pool_timeout=settings.ollama_pool_timeout_seconds,
        max_connections=settings.ollama_max_connections,
        max_keepalive_connections=settings.ollama_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry_seconds,
        max_concurrency=settings.ollama_max_concurrency,
        heavy_max_concurrency=settings.ollama_heavy_max_concurrency,
    )


def _get_shared_runtime(settings: Settings) -> _SharedOllamaRuntime:
    key = _runtime_key(settings)
    with _SHARED_RUNTIMES_LOCK:
        runtime = _SHARED_RUNTIMES.get(key)
        if runtime is None:
            runtime = _SharedOllamaRuntime(settings)
            _SHARED_RUNTIMES[key] = runtime
        return runtime


class OllamaClient:
    """Shared, bounded async Ollama HTTP client."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resource_manager: OllamaResourceManager | None = None,
    ) -> None:
        self.settings = settings
        if transport is None and resource_manager is None:
            self._runtime = _get_shared_runtime(settings)
        else:
            self._runtime = _SharedOllamaRuntime(
                settings,
                transport=transport,
                resource_manager=resource_manager,
            )
        self.resources = self._runtime.resources

    @property
    def runtime_identity(self) -> int:
        return id(self._runtime)

    async def start(self) -> None:
        await self._runtime.get_client()

    async def aclose(self) -> None:
        await self._runtime.close()

    def invalidate_model_cache(self) -> None:
        self._runtime.invalidate_models()

    async def health(self) -> dict[str, Any]:
        client = await self._runtime.get_client()
        started = monotonic()
        try:
            root = await client.get("/")
            root.raise_for_status()
            models = await self.list_models(force_refresh=True, allow_stale=False)
        except httpx.HTTPError as exc:
            metrics.inc("ollama.health_error")
            raise OllamaError(_http_error_message(exc)) from exc
        finally:
            metrics.observe("ollama.health_seconds", monotonic() - started)
        return {
            "status": "available",
            "root": root.text.strip(),
            "models": [item.get("name") or item.get("model") for item in models],
        }

    async def list_models(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        now = monotonic()
        cached = self._runtime.models_cache
        if (
            not force_refresh
            and cached is not None
            and now - self._runtime.models_cached_at
            <= self.settings.inventory_cache_ttl_seconds
        ):
            metrics.inc("ollama.models_cache_hit")
            return _copy_dict_list(cached)

        async with self._runtime.models_lock:
            now = monotonic()
            cached = self._runtime.models_cache
            if (
                not force_refresh
                and cached is not None
                and now - self._runtime.models_cached_at
                <= self.settings.inventory_cache_ttl_seconds
            ):
                metrics.inc("ollama.models_cache_hit")
                return _copy_dict_list(cached)

            active = client or await self._runtime.get_client()
            started = monotonic()
            try:
                response = await active.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise OllamaError("Ollama /api/tags returned a non-object payload.")
                models = [
                    dict(item)
                    for item in (payload.get("models") or [])
                    if isinstance(item, dict)
                ]
            except (httpx.HTTPError, ValueError, OllamaError) as exc:
                metrics.inc("ollama.models_error")
                if (
                    allow_stale
                    and cached is not None
                    and now - self._runtime.models_cached_at
                    <= self.settings.inventory_stale_if_error_seconds
                ):
                    metrics.inc("ollama.models_stale_cache_used")
                    return _copy_dict_list(cached)
                if isinstance(exc, OllamaError):
                    raise
                raise OllamaError(_http_error_message(exc)) from exc
            finally:
                metrics.observe("ollama.models_seconds", monotonic() - started)

            self._runtime.models_cache = models
            self._runtime.models_cached_at = monotonic()
            metrics.inc("ollama.models_refresh")
            return _copy_dict_list(models)

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        num_predict: int | None = None,
        response_format: StructuredFormat | None = None,
        think: bool | str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        payload = self._payload(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            num_predict=num_predict,
            response_format=response_format,
            think=think,
            options=options,
        )
        client = await self._runtime.get_client()
        started = monotonic()
        meter = get_current_execution_meter()
        if meter is not None:
            await meter.begin_model_attempt()
        success = False
        data: dict[str, Any] = {}
        metrics.inc("ollama.chat_requests")
        try:
            async with self.resources.acquire(model):
                log_kv(
                    logger,
                    logging.INFO,
                    "ollama_chat_request",
                    model=model,
                    messages=len(messages),
                    message_chars=sum(
                        len(str(item.get("content") or "")) for item in messages
                    ),
                    num_ctx=payload.get("options", {}).get("num_ctx"),
                    num_predict=payload.get("options", {}).get("num_predict"),
                    structured=bool(response_format is not None),
                    think=payload.get("think"),
                )
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                parsed = response.json()
                if not isinstance(parsed, dict):
                    raise OllamaError("Ollama /api/chat returned a non-object payload.")
                data = parsed
                if data.get("error"):
                    raise OllamaError(str(data["error"]))
                success = True
        except (httpx.HTTPError, ValueError, OllamaError) as exc:
            metrics.inc("ollama.chat_errors")
            if isinstance(exc, OllamaError):
                raise
            raise OllamaError(_http_error_message(exc)) from exc
        finally:
            metrics.observe("ollama.chat_seconds", monotonic() - started)
            if meter is not None:
                await meter.finish_model_attempt(
                    success=success,
                    prompt_tokens=_nonnegative_int(data.get("prompt_eval_count")),
                    generated_tokens=_nonnegative_int(data.get("eval_count")),
                    model_load_seconds=_nanoseconds_to_seconds(
                        data.get("load_duration")
                    ),
                )

        _record_ollama_telemetry(data)
        log_kv(
            logger,
            logging.INFO,
            "ollama_chat_response",
            model=model,
            status_code=response.status_code,
            elapsed_seconds=round(monotonic() - started, 3),
            load_duration=data.get("load_duration"),
            eval_count=data.get("eval_count"),
        )
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise OllamaError("Ollama /api/chat returned an invalid message object.")
        return LLMResponse(
            content=str(message.get("content") or ""),
            model=str(data.get("model") or model),
            raw=data,
        )

    async def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        num_predict: int | None = None,
        response_format: StructuredFormat | None = None,
        think: bool | str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            num_predict=num_predict,
            response_format=response_format,
            think=think,
            options=options,
        )
        client = await self._runtime.get_client()
        started = monotonic()
        final_payload: dict[str, Any] | None = None
        meter = get_current_execution_meter()
        if meter is not None:
            await meter.begin_model_attempt()
        success = False
        metrics.inc("ollama.stream_requests")
        try:
            async with self.resources.acquire(model):
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise OllamaError(
                                f"Invalid Ollama stream line: {line[:200]}"
                            ) from exc
                        if not isinstance(data, dict):
                            raise OllamaError(
                                "Ollama stream item was not a JSON object."
                            )
                        if data.get("error"):
                            raise OllamaError(str(data["error"]))
                        content = str((data.get("message") or {}).get("content") or "")
                        if content:
                            yield content
                        if data.get("done"):
                            final_payload = data
                            success = True
                            break
        except (httpx.HTTPError, OllamaError) as exc:
            metrics.inc("ollama.stream_errors")
            if isinstance(exc, OllamaError):
                raise
            raise OllamaError(_http_error_message(exc)) from exc
        finally:
            metrics.observe("ollama.stream_seconds", monotonic() - started)
            if meter is not None:
                data = final_payload or {}
                await meter.finish_model_attempt(
                    success=success,
                    prompt_tokens=_nonnegative_int(data.get("prompt_eval_count")),
                    generated_tokens=_nonnegative_int(data.get("eval_count")),
                    model_load_seconds=_nanoseconds_to_seconds(
                        data.get("load_duration")
                    ),
                )
        if final_payload is not None:
            _record_ollama_telemetry(final_payload)

    async def embed(self, *, model: str, text: str) -> list[float]:
        client = await self._runtime.get_client()
        started = monotonic()
        meter = get_current_execution_meter()
        if meter is not None:
            await meter.begin_model_attempt()
        success = False
        metrics.inc("ollama.embed_requests")
        try:
            async with self.resources.acquire(model):
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": model,
                        "input": text,
                        "keep_alive": self.settings.ollama_keep_alive,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise OllamaError(
                        "Ollama /api/embed returned a non-object payload."
                    )
                embeddings = data.get("embeddings") or []
                if not isinstance(embeddings, list) or not embeddings:
                    raise OllamaError("Ollama returned no embeddings.")
                vector = embeddings[0]
                if not isinstance(vector, list):
                    raise OllamaError("Ollama returned an invalid embedding vector.")
                success = True
                return [float(value) for value in vector]
        except (httpx.HTTPError, ValueError, TypeError, OllamaError) as exc:
            metrics.inc("ollama.embed_errors")
            if isinstance(exc, OllamaError):
                raise
            raise OllamaError(_http_error_message(exc)) from exc
        finally:
            metrics.observe("ollama.embed_seconds", monotonic() - started)
            if meter is not None:
                await meter.finish_model_attempt(success=success)

    async def _get_client(self) -> httpx.AsyncClient:
        return await self._runtime.get_client()

    def _payload(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        stream: bool,
        temperature: float | None,
        num_predict: int | None,
        response_format: StructuredFormat | None = None,
        think: bool | str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged_options: dict[str, Any] = {
            "temperature": (
                self.settings.ollama_temperature if temperature is None else temperature
            ),
            "num_predict": (
                self.settings.ollama_num_predict if num_predict is None else num_predict
            ),
            "num_ctx": int(getattr(self.settings, "ollama_num_ctx", 8192)),
        }
        if options:
            merged_options.update(dict(options))

        payload: dict[str, Any] = {
            "model": model,
            "stream": stream,
            "think": self.settings.ollama_think if think is None else think,
            "messages": [dict(message) for message in messages],
            "keep_alive": self.settings.ollama_keep_alive,
            "options": merged_options,
        }
        if response_format is not None:
            payload["format"] = (
                dict(response_format)
                if isinstance(response_format, Mapping)
                else response_format
            )
        return payload


def _copy_dict_list(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


def _http_error_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.strip().replace("\n", " ")[:500]
        detail = f"; response={body}" if body else ""
        return f"Ollama HTTP {exc.response.status_code}: {exc}{detail}"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _record_ollama_telemetry(data: Mapping[str, Any]) -> None:
    for field, metric_name in (
        ("total_duration", "ollama.total_duration_seconds"),
        ("load_duration", "ollama.load_duration_seconds"),
        ("prompt_eval_duration", "ollama.prompt_eval_duration_seconds"),
        ("eval_duration", "ollama.eval_duration_seconds"),
    ):
        value = data.get(field)
        if isinstance(value, (int, float)) and value >= 0:
            metrics.observe(metric_name, float(value) / 1_000_000_000.0)
    for field, metric_name in (
        ("prompt_eval_count", "ollama.prompt_tokens"),
        ("eval_count", "ollama.generated_tokens"),
    ):
        value = data.get(field)
        if isinstance(value, int) and value >= 0:
            metrics.observe(metric_name, float(value))


def _nonnegative_int(value: object) -> int:
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _nanoseconds_to_seconds(value: object) -> float:
    if isinstance(value, (int, float)) and value >= 0:
        return float(value) / 1_000_000_000.0
    return 0.0
