from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.ollama import OllamaClient

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


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

        json_schema = schema.model_json_schema()
        failures: list[str] = []
        models = self._structured_models()

        for attempt, model in enumerate(models, start=1):
            response = await self.ollama.chat(
                model=model,
                messages=self._structured_messages(
                    system=system,
                    payload=payload,
                    json_schema=json_schema,
                    retry=attempt > 1,
                ),
                temperature=0.0,
                response_format=json_schema,
                # Structured responses need final JSON, not a separate reasoning
                # trace. This also prevents thinking tokens from consuming the
                # complete num_predict budget before message.content is emitted.
                think=False,
            )

            content = response.content.strip()
            if not content:
                detail = self._empty_response_detail(response.raw)
                failures.append(f"{model!r}: empty final content ({detail})")
                logger.warning(
                    "structured_output_empty model=%s schema=%s attempt=%s detail=%s",
                    model,
                    schema.__name__,
                    attempt,
                    detail,
                )
                continue

            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                failures.append(f"{model!r}: {exc}")
                logger.warning(
                    "structured_output_invalid model=%s schema=%s attempt=%s",
                    model,
                    schema.__name__,
                    attempt,
                )

        failure_summary = " | ".join(failures) or "no structured response attempts"
        raise RuntimeError(
            f"Models returned invalid structured output for {schema.__name__}: "
            f"{failure_summary}"
        )

    async def invoke_text(self, *, system: str, payload: dict[str, Any]) -> str:
        if getattr(self.settings, "llm_backend", "echo") != "ollama":
            raise RuntimeError("LLM execution requires LLM_BACKEND=ollama")

        response = await self.ollama.chat(
            model=self.model,
            messages=self._messages(system, payload),
            temperature=getattr(self.settings, "ollama_temperature", 0.2),
        )
        return response.content

    def _structured_models(self) -> list[str]:
        """Return the primary model and at most one bounded JSON fallback.

        The general model is preferred as the fallback because it already serves
        normal structured planner/worker calls. ``model_fallback`` is used only
        when the general model is the primary model or is unavailable.
        """

        models = [self.model]
        for setting_name in ("model_general", "model_fallback"):
            candidate = str(getattr(self.settings, setting_name, "") or "").strip()
            if candidate and candidate not in models:
                models.append(candidate)
                break
        return models

    @staticmethod
    def _structured_messages(
        *,
        system: str,
        payload: dict[str, Any],
        json_schema: dict[str, Any],
        retry: bool,
    ) -> list[dict[str, str]]:
        request: dict[str, Any] = {
            "input": payload,
            "output_schema": json_schema,
            "response_rules": [
                "Return exactly one JSON object that validates against output_schema.",
                "Do not include markdown, commentary, or a reasoning trace.",
            ],
        }
        if retry:
            request["retry_instruction"] = (
                "The previous structured attempt was empty or invalid. "
                "Return only the final valid JSON object now."
            )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(request, ensure_ascii=False, default=str),
            },
        ]

    @staticmethod
    def _messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]

    @staticmethod
    def _empty_response_detail(raw: dict[str, Any]) -> str:
        message = raw.get("message") or {}
        thinking = message.get("thinking") if isinstance(message, dict) else ""
        thinking_chars = len(thinking) if isinstance(thinking, str) else 0
        return (
            f"done_reason={raw.get('done_reason')!r}, "
            f"eval_count={raw.get('eval_count')!r}, "
            f"thinking_chars={thinking_chars}"
        )
