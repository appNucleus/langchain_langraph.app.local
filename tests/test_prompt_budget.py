from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.base import StructuredAgent, StructuredOutputError


class WorkerLike(BaseModel):
    answer: str
    claims: list[dict[str, Any]] = []
    assumptions: list[str] = []
    missing_information: list[str] = []
    confidence: float = 0.5


class FakeOllama:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "llm_backend": "ollama",
        "ollama_model": "qwen3.5:4b",
        "ollama_temperature": 0.2,
        "ollama_num_ctx": 8192,
        "ollama_num_predict": 2048,
        "structured_output_reserve_tokens": 1536,
        "structured_prompt_chars_per_token": 3.0,
        "model_general": "qwen3.5:4b",
        "model_fallback": "granite3.3:8b",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(content: str, *, model: str, **raw: Any) -> SimpleNamespace:
    payload = {"model": model, "message": {"content": content}, **raw}
    return SimpleNamespace(content=content, model=model, raw=payload)


@pytest.mark.asyncio
async def test_large_research_payload_is_compacted_before_structured_call() -> None:
    client = FakeOllama(
        [
            _response(
                json.dumps(
                    {
                        "answer": "grounded",
                        "claims": [],
                        "assumptions": [],
                        "missing_information": [],
                        "confidence": 0.8,
                    }
                ),
                model="qwen3.5:9b",
                done_reason="stop",
                prompt_eval_count=3000,
                eval_count=40,
            )
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="qwen3.5:9b",
        ollama_client=client,  # type: ignore[arg-type]
    )
    payload = {
        "user_request": "current sports analysis",
        "evidence": [
            {"id": "e1", "source": "search", "content": "x" * 30000},
            {"id": "e2", "source": "news", "content": "y" * 30000},
        ],
        "history": [{"role": "assistant", "content": "z" * 8000}] * 10,
    }

    result = await agent.invoke_json(
        system="Use current evidence.", payload=payload, schema=WorkerLike
    )

    assert result.answer == "grounded"
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(json.dumps(sent["input"])) < len(json.dumps(payload))
    assert len(sent["input"]["history"]) <= 4
    assert client.calls[0]["think"] is False


@pytest.mark.asyncio
async def test_length_failure_retries_with_a_smaller_prompt() -> None:
    client = FakeOllama(
        [
            _response(
                "{",
                model="qwen3.5:9b",
                done_reason="length",
                prompt_eval_count=4095,
                eval_count=1,
            ),
            _response(
                (
                    '{"answer":"recovered","claims":[],"assumptions":[],'
                    '"missing_information":[],"confidence":0.7}'
                ),
                model="qwen3.5:4b",
                done_reason="stop",
                prompt_eval_count=2100,
                eval_count=50,
            ),
        ]
    )
    agent = StructuredAgent(
        _settings(ollama_num_ctx=4096),
        model="qwen3.5:9b",
        ollama_client=client,  # type: ignore[arg-type]
    )
    payload = {"evidence": [{"content": "x" * 40000}]}

    result = await agent.invoke_json(system="Return JSON.", payload=payload, schema=WorkerLike)

    assert result.answer == "recovered"
    first = client.calls[0]["messages"][1]["content"]
    second = client.calls[1]["messages"][1]["content"]
    assert len(second) < len(first)
    assert [call["model"] for call in client.calls] == ["qwen3.5:9b", "qwen3.5:4b"]


@pytest.mark.asyncio
async def test_two_failures_raise_typed_structured_error() -> None:
    client = FakeOllama(
        [
            _response("{", model="qwen3.5:9b", done_reason="length", prompt_eval_count=8191),
            _response("{", model="qwen3.5:4b", done_reason="length", prompt_eval_count=8191),
        ]
    )
    agent = StructuredAgent(
        _settings(),
        model="qwen3.5:9b",
        ollama_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await agent.invoke_json(
            system="Return JSON.",
            payload={"evidence": [{"content": "x" * 30000}]},
            schema=WorkerLike,
        )

    assert exc_info.value.schema_name == "WorkerLike"
    assert exc_info.value.attempted_models == ["qwen3.5:9b", "qwen3.5:4b"]
