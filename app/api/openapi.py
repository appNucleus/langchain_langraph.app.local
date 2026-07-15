from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Callable

from fastapi import FastAPI

from app.schemas.chat import (
    build_chat_request_openapi_examples,
    load_chat_request_example,
)


def openapi_with_chat_request_example(
    generated_openapi: Callable[[], dict[str, Any]],
    example_cache: dict[str, Any],
) -> dict[str, Any]:
    """Inject the documentation-only chat example lazily and once per app."""

    schema = generated_openapi()
    if "examples" not in example_cache:
        example_cache["examples"] = build_chat_request_openapi_examples(
            load_chat_request_example()
        )
    examples = example_cache["examples"]
    for path in ("/api/chat", "/api/chat/stream"):
        operation = (schema.get("paths") or {}).get(path, {}).get("post", {})
        content = (operation.get("requestBody") or {}).get("content", {})
        json_body = content.get("application/json")
        if isinstance(json_body, dict):
            json_body["examples"] = deepcopy(examples)
    return schema


def install_openapi_customization(app: FastAPI) -> None:
    """Install the existing lazy Swagger request-example customization."""

    generated_openapi = app.openapi
    chat_request_openapi_cache: dict[str, Any] = {}
    app.openapi = partial(  # type: ignore[method-assign]
        openapi_with_chat_request_example,
        generated_openapi,
        chat_request_openapi_cache,
    )
