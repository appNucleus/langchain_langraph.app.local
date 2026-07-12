from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    thread_id: str | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @field_validator("system_prompt", mode="before")
    @classmethod
    def normalize_optional_system_prompt(cls, value: Any) -> Any:
        """Treat omitted, null, and whitespace-only prompts equivalently."""

        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    backend: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        *,
        thread_id: str | None,
        response: str,
        backend: str,
        model: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChatResponse":
        return cls(
            thread_id=thread_id or str(uuid4()),
            response=response,
            backend=backend,
            model=model,
            metadata=metadata or {},
        )
