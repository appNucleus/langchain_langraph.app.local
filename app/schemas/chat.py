from __future__ import annotations

import json
import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

_EXAMPLE_REQUEST_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "docs" / "example_request"
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    thread_id: str | None = Field(
        default=None,
        description="Deprecated compatibility alias for conversation_id.",
    )
    conversation_id: str | None = None
    run_id: str | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_legacy_identity(self) -> "ChatRequest":
        if self.thread_id and self.conversation_id and self.thread_id != self.conversation_id:
            raise ValueError(
                "thread_id and conversation_id must match when both are provided"
            )
        if self.conversation_id is None and self.thread_id is not None:
            self.conversation_id = self.thread_id
        if self.conversation_id is None:
            self.conversation_id = str(uuid4())
        if self.thread_id is None:
            self.thread_id = self.conversation_id
        if self.run_id is None:
            self.run_id = str(uuid4())
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


@lru_cache(maxsize=None)
def _load_request_example(filename: str) -> dict[str, Any] | None:
    """Load one documentation-only Swagger request from the established folder.

    The file is used only while FastAPI constructs OpenAPI metadata. It never
    supplies Pydantic defaults, runtime request values, graph state, or tests.
    Missing or malformed documentation is non-fatal to the API runtime.
    """

    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".json"):
        logger.warning("openapi_request_example_invalid_filename: %s", filename)
        return None

    path = _EXAMPLE_REQUEST_DIRECTORY / safe_name
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("request example must be a JSON object")
        ChatRequest.model_validate(raw)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("openapi_request_example_unavailable file=%s error=%s", path, exc)
        return None
    return raw


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
    return {
        "default": {
            "summary": summary,
            "description": description,
            "value": deepcopy(value),
        }
    }
