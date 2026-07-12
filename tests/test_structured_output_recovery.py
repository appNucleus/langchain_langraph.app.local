from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.base import StructuredAgent


class VerificationLikeResult(BaseModel):
    verdict: str
    task_complete: bool
    issues: list[dict[str, Any]] = []
    required_actions: list[str] = []
    confidence: float


class FakeOllamaClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "llm_backend": "ollama",
        "ollama_model": "qwen3.5:4b",
        "ollama_temperature": 0.2,
        "model_general": "qwen3.5:4b",
        "model_fallback": "granite3.3:8b",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(content: str, *, model: str, **raw: Any) -> SimpleNamespace:
    payload = {"model": model, "message": {"content": content}, **raw}
    return SimpleNamespace(content=content, model=model, raw=payload)


@pytest.mark.asyncio
async def test_empty_reasoning_response_retries_once_with_general_model() -> None:
    client = FakeOllamaClient(
        [
            _response(
                "",
                model="deepseek-r1:8b",
                done_reason="length",
                eval_count=2048,
                message={"content": "", "thinking": "reasoning" * 20},
            ),
            _response(
                json.dumps(
                    {
                        "verdict": "research",
                        "task_complete": False,
                        "issues": [],
                        "required_actions": ["Retrieve current evidence"],
                        "confidence": 0.8,
                    }
                ),
                model="qwen3.5:4b",
                done_reason="stop",
                eval_count=80,
            ),
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="deepseek-r1:8b",
        ollama_client=client,  # type: ignore[arg-type]
    )

    result = await agent.invoke_json(
        system="Verify the result.",
        payload={"answer": "unknown"},
        schema=VerificationLikeResult,
    )

    assert result.verdict == "research"
    assert [call["model"] for call in client.calls] == [
        "deepseek-r1:8b",
        "qwen3.5:4b",
    ]
    assert all(call["think"] is False for call in client.calls)
    assert all(
        call["response_format"] == VerificationLikeResult.model_json_schema()
        for call in client.calls
    )
    retry_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert "retry_instruction" in retry_payload
    assert retry_payload["output_schema"] == VerificationLikeResult.model_json_schema()


@pytest.mark.asyncio
async def test_valid_primary_response_does_not_call_fallback() -> None:
    client = FakeOllamaClient(
        [
            _response(
                '{"verdict":"pass","task_complete":true,"issues":[],"required_actions":[],"confidence":0.9}',
                model="deepseek-r1:8b",
                done_reason="stop",
                eval_count=40,
            )
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="deepseek-r1:8b",
        ollama_client=client,  # type: ignore[arg-type]
    )

    result = await agent.invoke_json(
        system="Verify the result.",
        payload={"answer": "grounded"},
        schema=VerificationLikeResult,
    )

    assert result.task_complete is True
    assert len(client.calls) == 1
    assert client.calls[0]["think"] is False


@pytest.mark.asyncio
async def test_invalid_primary_json_uses_one_bounded_fallback() -> None:
    client = FakeOllamaClient(
        [
            _response("not-json", model="deepseek-r1:8b"),
            _response(
                '{"verdict":"revise","task_complete":false,"issues":[],"required_actions":[],"confidence":0.4}',
                model="qwen3.5:4b",
            ),
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="deepseek-r1:8b",
        ollama_client=client,  # type: ignore[arg-type]
    )

    result = await agent.invoke_json(
        system="Verify the result.",
        payload={},
        schema=VerificationLikeResult,
    )

    assert result.verdict == "revise"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_two_failed_attempts_raise_diagnostic_error() -> None:
    client = FakeOllamaClient(
        [
            _response(
                "",
                model="deepseek-r1:8b",
                done_reason="length",
                eval_count=2048,
                message={"content": "", "thinking": "x" * 100},
            ),
            _response("{}", model="qwen3.5:4b"),
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="deepseek-r1:8b",
        ollama_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError) as exc_info:
        await agent.invoke_json(
            system="Verify the result.",
            payload={},
            schema=VerificationLikeResult,
        )

    message = str(exc_info.value)
    assert "done_reason='length'" in message
    assert "eval_count=2048" in message
    assert "thinking_chars=100" in message
    assert "qwen3.5:4b" in message
    assert len(client.calls) == 2
