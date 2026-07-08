from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.logging_config import log_kv
from app.settings import Settings

logger = logging.getLogger(__name__)


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    raw: dict[str, Any]


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Small async Ollama HTTP client.

    We intentionally call Ollama's HTTP API directly instead of hiding it behind
    a heavyweight wrapper because newer thinking-capable models may emit
    `message.thinking`. This client always requests `think: false` by default
    and only returns `message.content` to the rest of the app.
    """

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    async def health(self) -> dict[str, Any]:
        async with self._client() as client:
            root = await client.get("/")
            root.raise_for_status()
            models = await self.list_models(client=client)
            return {
                "root": root.text.strip(),
                "models": [item.get("name") or item.get("model") for item in models],
            }

    async def list_models(self, *, client: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
        """Return live local Ollama models from GET /api/tags."""
        if client is not None:
            response = await client.get("/api/tags")
            response.raise_for_status()
            return list(response.json().get("models") or [])
        async with self._client() as owned_client:
            response = await owned_client.get("/api/tags")
            response.raise_for_status()
            return list(response.json().get("models") or [])

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
        log_kv(logger, logging.INFO, "ollama_chat_request", model=model, stream=False, messages=len(messages))
        async with self._client() as client:
            response = await client.post("/api/chat", json=payload)
            log_kv(logger, logging.INFO, "ollama_chat_response", model=model, status_code=response.status_code)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content") or ""
        return LLMResponse(content=content, model=data.get("model") or model, raw=data)

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
        log_kv(logger, logging.INFO, "ollama_stream_request", model=model, stream=True, messages=len(messages))
        async with self._client() as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                log_kv(logger, logging.INFO, "ollama_stream_response", model=model, status_code=response.status_code)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError(f"Invalid Ollama stream line: {line[:200]}") from exc
                    if data.get("error"):
                        raise OllamaError(str(data["error"]))
                    message = data.get("message") or {}
                    # Critical: never expose model thinking traces to callers.
                    content = message.get("content") or ""
                    if content:
                        yield content
                    if data.get("done"):
                        break

    async def embed(self, *, model: str, text: str) -> list[float]:
        payload = {"model": model, "input": text}
        async with self._client() as client:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise OllamaError("Ollama returned no embeddings.")
        return list(embeddings[0])

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(self.settings.ollama_timeout_seconds),
            transport=self._transport,
        )

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
                "temperature": self.settings.ollama_temperature if temperature is None else temperature,
                "num_predict": self.settings.ollama_num_predict if num_predict is None else num_predict,
            },
        }
