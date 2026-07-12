from __future__ import annotations

import copy
import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.ollama import OllamaClient
from app.logging_config import log_kv

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    """Raised after all bounded structured-output attempts have failed."""

    def __init__(
        self,
        *,
        schema_name: str,
        primary_model: str,
        attempted_models: list[str],
        failures: list[str],
    ) -> None:
        self.schema_name = schema_name
        self.primary_model = primary_model
        self.attempted_models = list(attempted_models)
        self.failures = list(failures)
        summary = " | ".join(failures) or "no structured response attempts"
        super().__init__(
            f"Models returned invalid structured output for {schema_name}: {summary}"
        )


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
        base_payload_budget = self._structured_payload_budget_chars(
            system=system,
            json_schema=json_schema,
        )
        original_payload_chars = self._json_chars(payload)

        for attempt, model in enumerate(models, start=1):
            # A fallback must not repeat an oversized prompt unchanged. The
            # second and final attempt uses a deliberately smaller payload.
            attempt_budget = (
                base_payload_budget
                if attempt == 1
                else max(1200, min(3000, int(base_payload_budget * 0.40)))
            )
            prepared_payload, compacted = self._compact_payload(
                payload,
                max_chars=attempt_budget,
            )
            messages = self._structured_messages(
                system=system,
                payload=prepared_payload,
                json_schema=json_schema,
                retry=attempt > 1,
            )
            message_chars = sum(len(item["content"]) for item in messages)
            estimated_tokens = self._estimate_tokens(message_chars)
            log_kv(
                logger,
                logging.INFO,
                "structured_prompt_prepared",
                schema=schema.__name__,
                model=model,
                attempt=attempt,
                original_payload_chars=original_payload_chars,
                prepared_payload_chars=self._json_chars(prepared_payload),
                payload_budget_chars=attempt_budget,
                message_chars=message_chars,
                estimated_prompt_tokens=estimated_tokens,
                num_ctx=int(getattr(self.settings, "ollama_num_ctx", 8192)),
                num_predict=int(getattr(self.settings, "ollama_num_predict", 2048)),
                compacted=compacted,
                evidence_items=self._list_length(prepared_payload.get("evidence")),
                history_messages=self._list_length(prepared_payload.get("history")),
            )
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
                messages=messages,
                temperature=0.0,
                response_format=json_schema,
                # Structured responses need final JSON, not a separate reasoning
                # trace. This prevents thinking tokens from consuming the output
                # allowance before message.content is emitted.
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
                self._log_fallback_if_available(
                    schema=schema,
                    models=models,
                    attempt=attempt,
                    failed_model=model,
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
                    context_exhausted=self._context_exhausted(response.raw),
                    **response_fields,
                )
                self._log_fallback_if_available(
                    schema=schema,
                    models=models,
                    attempt=attempt,
                    failed_model=model,
                    reason=(
                        "context_exhausted"
                        if self._context_exhausted(response.raw)
                        else "validation_error"
                    ),
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

        log_kv(
            logger,
            logging.ERROR,
            "structured_output_failed",
            schema=schema.__name__,
            primary_model=self.model,
            attempted_models=",".join(models),
            attempts=len(models),
        )
        raise StructuredOutputError(
            schema_name=schema.__name__,
            primary_model=self.model,
            attempted_models=models,
            failures=failures,
        )

    async def invoke_text(self, *, system: str, payload: dict[str, Any]) -> str:
        if getattr(self.settings, "llm_backend", "echo") != "ollama":
            raise RuntimeError("LLM execution requires LLM_BACKEND=ollama")

        num_ctx = int(getattr(self.settings, "ollama_num_ctx", 8192))
        reserve = int(getattr(self.settings, "structured_output_reserve_tokens", 1536))
        chars_per_token = float(
            getattr(self.settings, "structured_prompt_chars_per_token", 3.0)
        )
        payload_budget = max(
            2000,
            int((num_ctx - reserve) * chars_per_token) - len(system) - 1000,
        )
        prepared_payload, compacted = self._compact_payload(
            payload, max_chars=payload_budget
        )
        messages = self._messages(system, prepared_payload)
        log_kv(
            logger,
            logging.INFO,
            "text_prompt_prepared",
            model=self.model,
            original_payload_chars=self._json_chars(payload),
            prepared_payload_chars=self._json_chars(prepared_payload),
            message_chars=sum(len(item["content"]) for item in messages),
            payload_budget_chars=payload_budget,
            num_ctx=num_ctx,
            compacted=compacted,
        )
        response = await self.ollama.chat(
            model=self.model,
            messages=messages,
            temperature=getattr(self.settings, "ollama_temperature", 0.2),
        )
        return response.content

    def _structured_models(self) -> list[str]:
        """Return the primary model and at most one bounded JSON fallback."""

        models = [self.model]
        for setting_name in ("model_general", "model_fallback"):
            candidate = str(getattr(self.settings, setting_name, "") or "").strip()
            if candidate and candidate not in models:
                models.append(candidate)
                break
        return models

    def _structured_payload_budget_chars(
        self,
        *,
        system: str,
        json_schema: dict[str, Any],
    ) -> int:
        num_ctx = int(getattr(self.settings, "ollama_num_ctx", 8192))
        configured_reserve = int(
            getattr(self.settings, "structured_output_reserve_tokens", 1536)
        )
        reserve = min(max(256, configured_reserve), max(256, num_ctx // 2))
        chars_per_token = float(
            getattr(self.settings, "structured_prompt_chars_per_token", 3.0)
        )
        usable_chars = int(max(512, num_ctx - reserve) * chars_per_token)
        fixed_chars = (
            len(system)
            + len(json.dumps(json_schema, ensure_ascii=False, default=str))
            + 1800
        )
        return max(2000, usable_chars - fixed_chars)

    def _estimate_tokens(self, characters: int) -> int:
        chars_per_token = float(
            getattr(self.settings, "structured_prompt_chars_per_token", 3.0)
        )
        return max(1, int(characters / chars_per_token))

    @classmethod
    def _compact_payload(
        cls,
        payload: dict[str, Any],
        *,
        max_chars: int,
    ) -> tuple[dict[str, Any], bool]:
        original = copy.deepcopy(payload)
        if cls._json_chars(original) <= max_chars:
            return original, False

        # Stale history is context-only and should never crowd out the current
        # task, evidence, or JSON response allowance.
        history = original.get("history")
        if isinstance(history, list):
            original["history"] = history[-4:]

        for max_string_chars, max_items in (
            (4000, 12),
            (2500, 8),
            (1500, 6),
            (800, 5),
            (400, 4),
            (200, 3),
        ):
            candidate = cls._truncate_value(
                original,
                max_string_chars=max_string_chars,
                max_items=max_items,
            )
            if cls._json_chars(candidate) <= max_chars:
                return candidate, True

        final = cls._truncate_value(
            original,
            max_string_chars=120,
            max_items=2,
        )
        return final, True

    @classmethod
    def _truncate_value(
        cls,
        value: Any,
        *,
        max_string_chars: int,
        max_items: int,
    ) -> Any:
        if isinstance(value, str):
            if len(value) <= max_string_chars:
                return value
            return value[:max_string_chars] + "…[truncated]"
        if isinstance(value, list):
            items = value[-max_items:] if len(value) > max_items else value
            return [
                cls._truncate_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_items=max_items,
                )
                for item in items
            ]
        if isinstance(value, dict):
            return {
                str(key): cls._truncate_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_items=max_items,
                )
                for key, item in value.items()
            }
        return value

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
                "The previous structured attempt was empty, truncated, or invalid. "
                "Use the compact input and return only the final valid JSON object."
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
            f"prompt_eval_count={fields['prompt_eval_count']!r}, "
            f"eval_count={fields['eval_count']!r}, "
            f"thinking_chars={fields['thinking_chars']}"
        )

    def _context_exhausted(self, raw: dict[str, Any]) -> bool:
        if str(raw.get("done_reason") or "").lower() != "length":
            return False
        prompt_count = raw.get("prompt_eval_count")
        try:
            return int(prompt_count) >= int(getattr(self.settings, "ollama_num_ctx", 8192)) - 8
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _validation_summary(exc: ValidationError) -> str:
        parts: list[str] = []
        for error in exc.errors()[:5]:
            location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
            parts.append(f"{location}:{error.get('type', 'validation_error')}")
        return ",".join(parts)

    @staticmethod
    def _json_chars(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    def _list_length(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    @staticmethod
    def _log_fallback_if_available(
        *,
        schema: type[BaseModel],
        models: list[str],
        attempt: int,
        failed_model: str,
        reason: str,
    ) -> None:
        if attempt >= len(models):
            return
        log_kv(
            logger,
            logging.WARNING,
            "structured_output_fallback",
            schema=schema.__name__,
            failed_model=failed_model,
            fallback_model=models[attempt],
            reason=reason,
        )
