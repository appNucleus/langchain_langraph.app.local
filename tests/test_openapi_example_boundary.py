from __future__ import annotations

import ast
import json
from pathlib import Path

import app.schemas.chat as chat_schema
from app.schemas.chat import ChatRequest


def _clear_example_cache() -> None:
    chat_schema._load_request_example.cache_clear()


def test_missing_documentation_example_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chat_schema, "_EXAMPLE_REQUEST_DIRECTORY", tmp_path)
    _clear_example_cache()
    assert (
        chat_schema.load_chat_openapi_examples(
            "missing.json",
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
        "system_prompt": None,
        "metadata": {},
    }
    stream_value = {**chat_value, "message": "stream request"}
    (tmp_path / "chat.json").write_text(json.dumps(chat_value), encoding="utf-8")
    (tmp_path / "chat-stream.json").write_text(
        json.dumps(stream_value),
        encoding="utf-8",
    )
    monkeypatch.setattr(chat_schema, "_EXAMPLE_REQUEST_DIRECTORY", tmp_path)
    _clear_example_cache()

    normal = chat_schema.load_chat_openapi_examples(
        "chat.json",
        summary="Normal",
        description="Normal",
    )
    stream = chat_schema.load_chat_openapi_examples(
        "chat-stream.json",
        summary="Stream",
        description="Stream",
    )

    assert normal is not None
    assert stream is not None
    assert normal["default"]["value"]["message"] == "normal request"
    assert stream["default"]["value"]["message"] == "stream request"

    normal["default"]["value"]["message"] = "mutated"
    reread = chat_schema.load_chat_openapi_examples(
        "chat.json",
        summary="Normal",
        description="Normal",
    )
    assert reread is not None
    assert reread["default"]["value"]["message"] == "normal request"


def test_chat_request_has_no_documentation_derived_runtime_default() -> None:
    assert ChatRequest.model_fields["message"].is_required()
    assert ChatRequest.model_fields["metadata"].default_factory is dict


def test_factory_binds_one_json_file_to_each_post_route() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    factory = repository_root / "app" / "factory.py"
    tree = ast.parse(factory.read_text(encoding="utf-8"))
    loaded_files = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "load_chat_openapi_examples"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    openapi_sources = {
        ast.unparse(keyword.value)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Body"
        for keyword in call.keywords
        if keyword.arg == "openapi_examples"
    }

    assert loaded_files == {"chat.json", "chat-stream.json"}
    assert openapi_sources == {
        "deepcopy(chat_openapi_examples)",
        "deepcopy(chat_stream_openapi_examples)",
    }


def test_docs_example_directory_has_one_production_dependency() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    references: list[str] = []
    for path in (repository_root / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if '"docs" / "example_request"' in text:
            references.append(path.relative_to(repository_root).as_posix())
    assert references == ["app/schemas/chat.py"]
