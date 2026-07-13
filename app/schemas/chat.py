from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

CHAT_REQUEST_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "example_request"
    / "chat.json"
)
CHAT_REQUEST_COMPLETE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "example_request"
    / "chat-complete.json"
)

_FALLBACK_CHAT_REQUEST_EXAMPLE: dict[str, Any] = {
    "message": "Continue the analysis",
}


class ChatRequest(BaseModel):
    """Public chat request with backward-compatible identity fields."""

    message: str = Field(..., min_length=1, max_length=20000)
    thread_id: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Backward-compatible alias for conversation_id. Omit both IDs to start "
            "a new conversation and let the server generate a GUID."
        ),
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=200,
        description="Stable user-visible conversation identifier.",
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Optional UUID for an idempotent retry. Omit it for a new server-generated run."
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


def _validated_chat_request_example(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return deepcopy(_FALLBACK_CHAT_REQUEST_EXAMPLE)
    try:
        request = ChatRequest.model_validate(raw)
    except ValidationError:
        return deepcopy(_FALLBACK_CHAT_REQUEST_EXAMPLE)
    return request.model_dump(mode="json", exclude_unset=True)


def load_chat_request_example(path: Path | None = None) -> dict[str, Any]:
    """Load the documentation-only OpenAPI request example from ``chat.json``.

    This mapping decorates the generated OpenAPI schema only. It does not define
    ``ChatRequest`` defaults and is never used as chat execution input.
    """

    example_path = path or CHAT_REQUEST_EXAMPLE_PATH
    try:
        raw = json.loads(example_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(_FALLBACK_CHAT_REQUEST_EXAMPLE)
    return _validated_chat_request_example(raw)


def build_chat_request_openapi_examples(
    example: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a defensive OpenAPI example object from one validated request mapping."""

    return {
        "default": {
            "summary": "Configured chat request example",
            "description": (
                "Documentation-only request body loaded lazily from "
                "docs/example_request/chat.json when OpenAPI is generated. "
                "It does not define request-model defaults or runtime execution."
            ),
            "value": deepcopy(dict(example)),
        }
    }


class ChatResponse(BaseModel):
    thread_id: str
    conversation_id: str
    run_id: str | None = None
    response: str
    backend: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        *,
        thread_id: str | None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        response: str,
        backend: str,
        model: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatResponse":
        resolved_conversation_id = conversation_id or thread_id or str(uuid4())
        return cls(
            thread_id=resolved_conversation_id,
            conversation_id=resolved_conversation_id,
            run_id=run_id,
            response=response,
            backend=backend,
            model=model,
            metadata=metadata or {},
        )
