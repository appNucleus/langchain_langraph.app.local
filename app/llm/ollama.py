from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from app.llm.errors import OllamaError
from app.llm.resource_manager import OllamaResourceManager
from app.logging_config import log_kv
from app.observability.timing import measure
from app.orchestration.execution_meter import get_current_execution_meter
from app.settings import Settings

logger = logging.getLogger(__name__)
ChatMessage = dict[str, str]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    raw: dict[str, Any]


class OllamaClient:
    """Shared, bounded async Ollama HTTP client with physical-attempt metering."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resource_manager: OllamaResourceManager | None = None,
    ) -> None:
        self.settings = settings
        self._custom_transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self.resources = resource_manager or OllamaResourceManager(settings)

    async def start(self) -> None:
        await self._get_client()

    async def aclose(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def health(self) -> dict[str, Any]:
        client = await self._get_client()
        root = await client.get("/")
        root.raise_for_status()
        models = await self.list_models()
        return {
            "root": root.text.strip(),
            "models": [item.get("name") or item.get("model") for item in models],
        }

    async def list_models(
        self, *, client: httpx.AsyncClient | None = None
    ) -> list[dict[str, Any]]:
        active = client or await self._get_client()
        response = await active.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        return [item for item in (payload.get("models") or []) if isinstance(item, dict)]

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> LLMResponse:
        payload = self._payload(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            num_predict=num_predict,
        )
        client = await self._get_client()
        meter = get_current_execution_meter()
        async with _acquire_model(self.resources, model, meter):
            if meter is not None:
                await meter.begin_model_attempt()
            success = False
            data: dict[str, Any] = {}
            try:
                with measure() as timer:
                    log_kv(
                        logger,
                        logging.INFO,
                        "ollama_chat_request",
                        model=model,
                        messages=len(messages),
                    )
                    if meter is None:
                        response = await client.post("/api/chat", json=payload)
                    else:
                        async with asyncio.timeout(
                            max(0.001, meter.remaining_seconds())
                        ):
                            response = await client.post("/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    log_kv(
                        logger,
                        logging.INFO,
                        "ollama_chat_response",
                        model=model,
                        status_code=response.status_code,
                        elapsed_seconds=round(timer.elapsed_seconds, 3),
                        load_duration=data.get("load_duration"),
                        eval_count=data.get("eval_count"),
                    )
                if data.get("error"):
                    raise OllamaError(str(data["error"]))
                message = data.get("message") or {}
                success = True
                return LLMResponse(
                    content=str(message.get("content") or ""),
                    model=str(data.get("model") or model),
                    raw=data,
                )
            except asyncio.CancelledError:
                if meter is not None:
                    meter.record_cancellation()
                raise
            finally:
                if meter is not None:
                    await meter.finish_model_attempt(
                        success=success,
                        prompt_tokens=_integer(data.get("prompt_eval_count")),
                        generated_tokens=_integer(data.get("eval_count")),
                        model_load_seconds=_nanoseconds_to_seconds(
                            data.get("load_duration")
                        ),
                    )

    async def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            num_predict=num_predict,
        )
        client = await self._get_client()
        meter = get_current_execution_meter()
        async with _acquire_model(self.resources, model, meter):
            if meter is not None:
                await meter.begin_model_attempt()
            request_started = time.monotonic()
            first_token_at: float | None = None
            success = False
            final_data: dict[str, Any] = {}
            try:
                async with _stream_with_deadline(
                    client, payload, meter
                ) as response:
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
                        if data.get("error"):
                            raise OllamaError(str(data["error"]))
                        final_data = data
                        content = str((data.get("message") or {}).get("content") or "")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            yield content
                        if data.get("done"):
                            success = True
                            break
            except asyncio.CancelledError:
                if meter is not None:
                    meter.record_cancellation()
                raise
            finally:
                if meter is not None:
                    ttft = (
                        first_token_at - request_started
                        if first_token_at is not None
                        else None
                    )
                    await meter.finish_model_attempt(
                        success=success,
                        prompt_tokens=_integer(final_data.get("prompt_eval_count")),
                        generated_tokens=_integer(final_data.get("eval_count")),
                        model_load_seconds=_nanoseconds_to_seconds(
                            final_data.get("load_duration")
                        ),
                        time_to_first_token=ttft,
                    )

    async def embed(self, *, model: str, text: str) -> list[float]:
        client = await self._get_client()
        meter = get_current_execution_meter()
        async with _acquire_model(self.resources, model, meter):
            if meter is not None:
                await meter.begin_model_attempt()
            success = False
            data: dict[str, Any] = {}
            try:
                if meter is None:
                    response = await client.post(
                        "/api/embed", json={"model": model, "input": text}
                    )
                else:
                    async with asyncio.timeout(
                        max(0.001, meter.remaining_seconds())
                    ):
                        response = await client.post(
                            "/api/embed", json={"model": model, "input": text}
                        )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings") or []
                if not embeddings:
                    raise OllamaError("Ollama returned no embeddings.")
                success = True
                return list(embeddings[0])
            except asyncio.CancelledError:
                if meter is not None:
                    meter.record_cancellation()
                raise
            finally:
                if meter is not None:
                    await meter.finish_model_attempt(
                        success=success,
                        prompt_tokens=_integer(data.get("prompt_eval_count")),
                        model_load_seconds=_nanoseconds_to_seconds(
                            data.get("load_duration")
                        ),
                    )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
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
                self._client = httpx.AsyncClient(
                    base_url=self.settings.ollama_base_url.rstrip("/"),
                    timeout=timeout,
                    transport=self._custom_transport,
                    limits=limits,
                )
            return self._client

    def _payload(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        stream: bool,
        temperature: float | None,
        num_predict: int | None,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "stream": stream,
            "think": bool(self.settings.ollama_think),
            "messages": list(messages),
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": (
                    self.settings.ollama_temperature
                    if temperature is None
                    else temperature
                ),
                "num_predict": (
                    self.settings.ollama_num_predict
                    if num_predict is None
                    else num_predict
                ),
            },
        }


@asynccontextmanager
async def _acquire_model(resources: Any, model: str, meter: Any):
    """Bound queue wait and execution by the request's absolute deadline."""

    wait_started = time.monotonic()
    if meter is None:
        async with resources.acquire(model):
            yield
        return
    try:
        async with asyncio.timeout(max(0.001, meter.remaining_seconds())):
            async with resources.acquire(model):
                meter.add_queue_wait(time.monotonic() - wait_started)
                yield
    except TimeoutError:
        meter.add_queue_wait(time.monotonic() - wait_started)
        raise


@asynccontextmanager
async def _stream_with_deadline(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    meter: Any,
):
    if meter is None:
        async with client.stream("POST", "/api/chat", json=payload) as response:
            yield response
        return
    async with asyncio.timeout(max(0.001, meter.remaining_seconds())):
        async with client.stream("POST", "/api/chat", json=payload) as response:
            yield response


def _integer(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _nanoseconds_to_seconds(value: object) -> float:
    try:
        return max(0.0, float(value or 0.0) / 1_000_000_000.0)
    except (TypeError, ValueError):
        return 0.0
