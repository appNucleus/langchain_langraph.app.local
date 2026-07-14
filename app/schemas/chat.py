from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

CHAT_REQUEST_EXAMPLE_FILENAME = "chat.json"
CHAT_STREAM_REQUEST_EXAMPLE_FILENAME = "chat-stream.json"

_EXAMPLE_REQUEST_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "docs" / "example_request"
)


class ChatRequest(BaseModel):
    """Public chat request with server-owned execution identity generation."""

    message: str = Field(..., min_length=1, max_length=20000)
    thread_id: str | None = Field(
        default=None,
        max_length=200,
        description="Deprecated compatibility alias for conversation_id.",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=200,
        description="Stable user-visible conversation identifier.",
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Optional UUID for an idempotent retry. Omit it for a new "
            "server-generated run."
        ),
    )
    resume: bool = Field(
        default=False,
        description="Resume an interrupted run using a server-issued resume_token.",
    )
    resume_token: str | None = Field(
        default=None,
        max_length=4096,
        description="Signed token returned by the server for explicit run resumption.",
    )
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @field_validator(
        "thread_id",
        "conversation_id",
        "run_id",
        "resume_token",
        "system_prompt",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("run_id must be a valid UUID") from exc

    @model_validator(mode="after")
    def validate_identity_contract(self) -> "ChatRequest":
        if (
            self.thread_id
            and self.conversation_id
            and self.thread_id != self.conversation_id
        ):
            raise ValueError(
                "thread_id and conversation_id must match when both are provided"
            )
        if self.resume and not self.resume_token:
            raise ValueError("resume_token is required when resume=true")
        if self.resume_token and not self.resume:
            raise ValueError("resume must be true when resume_token is provided")
        return self


class ChatResponse(BaseModel):
    thread_id: str
    conversation_id: str | None = None
    run_id: str | None = None
    execution_thread_id: str | None = None
    response: str
    backend: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_compatibility_identity(self) -> "ChatResponse":
        self.conversation_id = self.conversation_id or self.thread_id
        self.run_id = self.run_id or str(uuid4())
        self.execution_thread_id = (
            self.execution_thread_id or f"{self.conversation_id}:{self.run_id}"
        )
        return self

    @classmethod
    def from_result(
        cls,
        *,
        thread_id: str | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        execution_thread_id: str | None = None,
        response: str,
        backend: str,
        model: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatResponse":
        resolved_conversation_id = conversation_id or thread_id or str(uuid4())
        resolved_run_id = run_id or str(uuid4())
        resolved_execution_thread_id = (
            execution_thread_id or f"{resolved_conversation_id}:{resolved_run_id}"
        )
        return cls(
            thread_id=resolved_conversation_id,
            conversation_id=resolved_conversation_id,
            run_id=resolved_run_id,
            execution_thread_id=resolved_execution_thread_id,
            response=response,
            backend=backend,
            model=model,
            metadata=metadata or {},
        )


def _validate_request_example(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate documentation data without turning it into request defaults."""

    raw = deepcopy(dict(value))
    ChatRequest.model_validate(raw)
    return raw


def _read_request_example(filename: str) -> dict[str, Any] | None:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".json"):
        logger.warning("openapi_request_example_invalid_filename: %s", filename)
        return None

    path = _EXAMPLE_REQUEST_DIRECTORY / safe_name
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("request example must be a JSON object")
        return _validate_request_example(raw)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "openapi_request_example_unavailable file=%s error=%s",
            path,
            exc,
        )
        return None


@lru_cache(maxsize=None)
def _load_request_example(filename: str) -> dict[str, Any] | None:
    """Cache documentation examples used by generated OpenAPI metadata."""

    return _read_request_example(filename)


def _build_openapi_examples(
    value: Mapping[str, Any],
    *,
    summary: str,
    description: str,
) -> dict[str, dict[str, Any]]:
    validated = _validate_request_example(value)
    return {
        "default": {
            "summary": summary,
            "description": description,
            "value": deepcopy(validated),
        }
    }


def load_chat_openapi_examples(
    filename: str,
    *,
    summary: str,
    description: str,
) -> dict[str, dict[str, Any]] | None:
    """Return a defensive OpenAPI mapping for one POST request JSON file."""

    value = _load_request_example(filename)
    if value is None:
        return None
    return _build_openapi_examples(
        value,
        summary=summary,
        description=description,
    )


def load_chat_request_example(
    filename: str = CHAT_REQUEST_EXAMPLE_FILENAME,
) -> dict[str, Any] | None:
    """Read one documentation-only request example without sharing cache state."""

    value = _read_request_example(filename)
    return deepcopy(value) if value is not None else None


def load_request_example(
    filename: str = CHAT_REQUEST_EXAMPLE_FILENAME,
) -> dict[str, Any] | None:
    """Compatibility alias for the established documentation loader."""

    return load_chat_request_example(filename)


def build_chat_request_openapi_examples(
    example: Mapping[str, Any] | str | None = None,
    *,
    summary: str = "Complete chat request",
    description: str = "Default values for the chat request.",
) -> dict[str, dict[str, Any]] | None:
    if isinstance(example, Mapping):
        return _build_openapi_examples(
            example,
            summary=summary,
            description=description,
        )
    filename = example or CHAT_REQUEST_EXAMPLE_FILENAME
    return load_chat_openapi_examples(
        filename,
        summary=summary,
        description=description,
    )


def build_chat_stream_request_openapi_examples(
    example: Mapping[str, Any] | str | None = None,
    *,
    summary: str = "Complete streaming chat request",
    description: str = "Default values for the streaming chat request.",
) -> dict[str, dict[str, Any]] | None:
    if isinstance(example, Mapping):
        return _build_openapi_examples(
            example,
            summary=summary,
            description=description,
        )
    filename = example or CHAT_STREAM_REQUEST_EXAMPLE_FILENAME
    return load_chat_openapi_examples(
        filename,
        summary=summary,
        description=description,
    )
