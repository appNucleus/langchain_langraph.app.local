from __future__ import annotations
import json
from typing import Any, TypeVar, Type
from pydantic import BaseModel
from langchain_ollama import ChatOllama

T = TypeVar('T', bound=BaseModel)

class StructuredAgent:
    def __init__(self, settings: Any, model: str | None = None) -> None:
        self.settings = settings
        self.model = model or getattr(settings, 'ollama_model', 'qwen2.5:0.5b')

    async def invoke_json(self, *, system: str, payload: dict[str, Any], schema: Type[T]) -> T:
        if getattr(self.settings, 'llm_backend', 'echo') != 'ollama':
            raise RuntimeError('Structured LLM execution requires LLM_BACKEND=ollama')
        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=self.model,
            temperature=0,
            timeout=self.settings.ollama_timeout_seconds,
            format='json',
        )
        result = await llm.ainvoke([('system', system), ('human', json.dumps(payload, ensure_ascii=False))])
        content = result.content if isinstance(result.content, str) else str(result.content)
        return schema.model_validate_json(content)

    async def invoke_text(self, *, system: str, payload: dict[str, Any]) -> str:
        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=self.model,
            temperature=getattr(self.settings, 'ollama_temperature', 0.2),
            timeout=self.settings.ollama_timeout_seconds,
        )
        result = await llm.ainvoke([('system', system), ('human', json.dumps(payload, ensure_ascii=False))])
        return result.content if isinstance(result.content, str) else str(result.content)
