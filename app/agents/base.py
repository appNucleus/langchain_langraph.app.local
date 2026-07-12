from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.ollama import OllamaClient
from app.logging_config import log_kv

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
            log_kv(
                logger,
                logging.INFO,
                "structured_output_attempt",
                schema=schema.__name__,
                model=model,
                primary_model=self.model,
                attempt=attempt,
                max_attempts=len(models),
                fallback=attempt > 1,
            )
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
            response_fields = self._response_log_fields(response.raw, content)
            if not content:
                detail = self._empty_response_detail(response.raw)
                failures.append(f"{model!r}: empty final content ({detail})")
                log_kv(
                    logger,
                    logging.WARNING,
                    "structured_output_empty",
                    schema=schema.__name__,
                    model=model,
                    attempt=attempt,
                    max_attempts=len(models),
                    **response_fields,
                )
                if attempt < len(models):
                    log_kv(
                        logger,
                        logging.WARNING,
                        "structured_output_fallback",
                        schema=schema.__name__,
                        failed_model=model,
                        fallback_model=models[attempt],
                        reason="empty_content",
                    )
                continue

            try:
                result = schema.model_validate_json(content)
            except ValidationError as exc:
                summary = self._validation_summary(exc)
                failures.append(f"{model!r}: {exc}")
                log_kv(
                    logger,
                    logging.WARNING,
                    "structured_output_invalid",
                    schema=schema.__name__,
                    model=model,
                    attempt=attempt,
                    max_attempts=len(models),
                    validation_errors=len(exc.errors()),
                    validation_summary=summary,
                    **response_fields,
                )
                if attempt < len(models):
                    log_kv(
                        logger,
                        logging.WARNING,
                        "structured_output_fallback",
                        schema=schema.__name__,
                        failed_model=model,
                        fallback_model=models[attempt],
                        reason="validation_error",
                    )
                continue

            log_kv(
                logger,
                logging.INFO,
                "structured_output_valid",
                schema=schema.__name__,
                model=model,
                attempt=attempt,
                fallback_used=attempt > 1,
                **response_fields,
            )
            return result

        failure_summary = " | ".join(failures) or "no structured response attempts"
        log_kv(
            logger,
            logging.ERROR,
            "structured_output_failed",
            schema=schema.__name__,
            primary_model=self.model,
            attempted_models=",".join(models),
            attempts=len(models),
        )
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
    def _response_log_fields(raw: dict[str, Any], content: str) -> dict[str, object]:
        message = raw.get("message") or {}
        thinking = message.get("thinking") if isinstance(message, dict) else ""
        return {
            "content_chars": len(content),
            "thinking_chars": len(thinking) if isinstance(thinking, str) else 0,
            "done_reason": raw.get("done_reason"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "load_duration": raw.get("load_duration"),
            "total_duration": raw.get("total_duration"),
        }

    @staticmethod
    def _empty_response_detail(raw: dict[str, Any]) -> str:
        fields = StructuredAgent._response_log_fields(raw, "")
        return (
            f"done_reason={fields['done_reason']!r}, "
            f"eval_count={fields['eval_count']!r}, "
            f"thinking_chars={fields['thinking_chars']}"
        )

    @staticmethod
    def _validation_summary(exc: ValidationError) -> str:
        parts: list[str] = []
        for error in exc.errors()[:5]:
            location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
            parts.append(f"{location}:{error.get('type', 'validation_error')}")
        return ",".join(parts)
