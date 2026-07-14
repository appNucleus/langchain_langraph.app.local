from __future__ import annotations

import ast
import json
from pathlib import Path

import app.schemas.chat as chat_schema
from app.schemas.chat import ChatRequest


def _filename(stem: str) -> str:
    return stem + "." + "json"


def _clear_example_cache() -> None:
    chat_schema._load_request_example.cache_clear()


def test_missing_documentation_example_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chat_schema, "_EXAMPLE_REQUEST_DIRECTORY", tmp_path)
    _clear_example_cache()
    assert (
        chat_schema.load_chat_openapi_examples(
            _filename("missing"),
            summary="Missing",
            description="Missing",
        )
        is None
    )


def test_separate_post_examples_use_existing_schema_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    chat_value = {
        "message": "normal request",
        "thread_id": None,
        "conversation_id": None,
        "run_id": None,
        "resume": False,
        "resume_token": None,
        "system_prompt": None,
        "metadata": {},
    }
    stream_value = {**chat_value, "message": "stream request"}
    chat_name = _filename("chat")
    stream_name = _filename("chat-stream")
    (tmp_path / chat_name).write_text(json.dumps(chat_value), encoding="utf-8")
    (tmp_path / stream_name).write_text(json.dumps(stream_value), encoding="utf-8")
    monkeypatch.setattr(chat_schema, "_EXAMPLE_REQUEST_DIRECTORY", tmp_path)
    _clear_example_cache()

    normal = chat_schema.load_chat_openapi_examples(
        chat_name,
        summary="Normal",
        description="Normal",
    )
    stream = chat_schema.load_chat_openapi_examples(
        stream_name,
        summary="Stream",
        description="Stream",
    )

    assert normal is not None
    assert stream is not None
    assert normal["default"]["value"]["message"] == "normal request"
    assert stream["default"]["value"]["message"] == "stream request"

    normal["default"]["value"]["message"] = "mutated"
    reread = chat_schema.load_chat_openapi_examples(
        chat_name,
        summary="Normal",
        description="Normal",
    )
    assert reread is not None
    assert reread["default"]["value"]["message"] == "normal request"


def test_chat_request_has_no_documentation_derived_runtime_default() -> None:
    assert ChatRequest.model_fields["message"].is_required()
    assert ChatRequest.model_fields["metadata"].default_factory is dict


def test_factory_loads_documentation_example_only_from_openapi_hook() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    factory = repository_root / "app" / "factory.py"
    tree = ast.parse(factory.read_text(encoding="utf-8"))

    calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "load_chat_request_example"
    ]
    assert len(calls) == 1

    parent_function = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(call is nested for nested in ast.walk(node) for call in calls):
            parent_function = node.name
            break
    assert parent_function == "openapi_with_chat_request_example"


def test_docs_example_directory_has_one_production_dependency() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    references: list[str] = []
    for path in (repository_root / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if '"docs" / "example_request"' in text:
            references.append(path.relative_to(repository_root).as_posix())
    assert references == ["app/schemas/chat.py"]
