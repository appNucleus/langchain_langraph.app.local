from __future__ import annotations

from copy import deepcopy
from functools import partial
from inspect import Signature, signature
from typing import Any, Callable

from fastapi import FastAPI

from app.schemas.chat import (
    CHAT_REQUEST_EXAMPLE_FILENAME,
    CHAT_STREAM_REQUEST_EXAMPLE_FILENAME,
    build_chat_request_openapi_examples,
    build_chat_stream_request_openapi_examples,
    load_chat_request_example,
)

RequestExampleLoader = Callable[..., dict[str, Any] | None]


def _accepts_filename(loader: RequestExampleLoader) -> bool:
    try:
        loader_signature: Signature = signature(loader)
        loader_signature.bind(CHAT_REQUEST_EXAMPLE_FILENAME)
    except (TypeError, ValueError):
        return False
    return True


def _load_request_examples(
    loader: RequestExampleLoader,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if _accepts_filename(loader):
        return (
            loader(CHAT_REQUEST_EXAMPLE_FILENAME),
            loader(CHAT_STREAM_REQUEST_EXAMPLE_FILENAME),
        )

    shared = loader()
    return shared, deepcopy(shared)


def _build_examples(
    value: dict[str, Any] | None,
    builder: Callable[[dict[str, Any]], dict[str, dict[str, Any]] | None],
) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    return builder(value)


def openapi_with_chat_request_example(
    generated_openapi: Callable[[], dict[str, Any]],
    example_cache: dict[str, Any],
    *,
    request_example_loader: RequestExampleLoader | None = None,
) -> dict[str, Any]:
    """Inject documentation-only examples lazily and once per application."""

    schema = generated_openapi()
    if "chat_examples" not in example_cache:
        chat_value, stream_value = _load_request_examples(
            request_example_loader or load_chat_request_example
        )
        example_cache["chat_examples"] = _build_examples(
            chat_value,
            build_chat_request_openapi_examples,
        )
        example_cache["stream_examples"] = _build_examples(
            stream_value,
            build_chat_stream_request_openapi_examples,
        )

    examples_by_path = {
        "/api/chat": example_cache["chat_examples"],
        "/api/chat/stream": example_cache["stream_examples"],
    }
    for path, examples in examples_by_path.items():
        operation = (schema.get("paths") or {}).get(path, {}).get("post", {})
        content = (operation.get("requestBody") or {}).get("content", {})
        json_body = content.get("application/json")
        if isinstance(json_body, dict) and examples is not None:
            json_body["examples"] = deepcopy(examples)
    return schema


def install_openapi_customization(
    app: FastAPI,
    *,
    request_example_loader: RequestExampleLoader | None = None,
) -> None:
    """Install the lazy Swagger request-example customization."""

    generated_openapi = app.openapi
    chat_request_openapi_cache: dict[str, Any] = {}
    app.openapi = partial(  # type: ignore[method-assign]
        openapi_with_chat_request_example,
        generated_openapi,
        chat_request_openapi_cache,
        request_example_loader=request_example_loader,
    )
