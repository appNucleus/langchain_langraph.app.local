from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.ollama import OllamaClient

T = TypeVar("T", bound=BaseModel)


class StructuredAgent:
    """Role-agent base that uses the process-shared Ollama runtime."""

    def __init__(
        self,
        settings: Any,
        model: str | None = None,
        *,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or getattr(settings, "ollama_model", "qwen2.5:0.5b")
        self.ollama = ollama_client or OllamaClient(settings)

    async def invoke_json(
        self,
        *,
        system: str,
        payload: dict[str, Any],
        schema: type[T],
    ) -> T:
        if getattr(self.settings, "llm_backend", "echo") != "ollama":
            raise RuntimeError("Structured LLM execution requires LLM_BACKEND=ollama")

        response = await self.ollama.chat(
            model=self.model,
            messages=self._messages(system, payload),
            temperature=0.0,
            response_format=schema.model_json_schema(),
        )
        try:
            return schema.model_validate_json(response.content)
        except ValidationError as exc:
            raise RuntimeError(
                f"Model {self.model!r} returned invalid structured output for "
                f"{schema.__name__}: {exc}"
            ) from exc

    async def invoke_text(self, *, system: str, payload: dict[str, Any]) -> str:
        if getattr(self.settings, "llm_backend", "echo") != "ollama":
            raise RuntimeError("LLM execution requires LLM_BACKEND=ollama")

        response = await self.ollama.chat(
            model=self.model,
            messages=self._messages(system, payload),
            temperature=getattr(self.settings, "ollama_temperature", 0.2),
        )
        return response.content

    @staticmethod
    def _messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
