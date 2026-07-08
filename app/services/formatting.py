from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.prompts.chat import ChatPromptTemplate

from app.mcp.client import MCPToolResult
from app.services.memory import ConversationTurn, InMemoryConversationStore
from app.services.routing import QueryPlan
from app.settings import Settings


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
THINKING_PROCESS_RE = re.compile(
    r"^\s*(Thinking Process|Thought Process)\s*:.*?(?=\n\s*\S|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


class PromptBuilder:
    """Build final LangChain chat prompts for the answering model.

    The app uses LangGraph for orchestration and LangChain Core prompt objects
    for message construction. The final transport to Ollama is intentionally a
    thin HTTP client so we can force `think: false` and never expose thinking
    traces returned by newer local models.
    """

    def __init__(self, settings: Settings, memory: InMemoryConversationStore) -> None:
        self.settings = settings
        self.memory = memory
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                ("user", "{user}"),
            ]
        )

    def build_messages(
        self,
        *,
        user_message: str,
        system_prompt: str,
        history: Sequence[ConversationTurn],
        plan: QueryPlan,
        rewritten_query: str | None,
        tool_results: Sequence[MCPToolResult],
        references: Sequence[dict[str, str]],
    ) -> list[dict[str, str]]:
        evidence = render_tool_evidence(tool_results, max_chars=self.settings.max_tool_chars)
        ref_text = render_reference_instruction(references)
        history_text = self.memory.render(history)
        system = (
            f"{system_prompt}\n\n"
            "Operational rules:\n"
            "- Answer the user's latest message, using concise session context only when relevant.\n"
            "- If tool evidence is present, base factual/current claims on that evidence.\n"
            "- If evidence is missing or weak, say what is missing; do not invent sources.\n"
            "- Never expose chain-of-thought, hidden reasoning, or fields named thinking.\n"
            "- Prefer a direct answer first, then relevant details.\n"
            "- For current information, mention the freshness/limits of the evidence when useful.\n"
            "- Include references naturally when source URLs/titles are available.\n"
        )
        user = (
            f"Current session context:\n{history_text}\n\n"
            f"Routing decision: intent={plan.intent}; model_key={plan.model_key}; tools={plan.tools}; reason={plan.reason}\n"
            f"Optimized search query: {rewritten_query or 'not needed'}\n\n"
            f"Tool evidence:\n{evidence}\n\n"
            f"Reference guidance:\n{ref_text}\n\n"
            f"User message:\n{user_message}"
        )
        value = self._prompt.invoke({"system": system, "user": user})
        messages: list[dict[str, str]] = []
        for msg in value.messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            messages.append({"role": role, "content": str(msg.content)})
        return messages


def render_tool_evidence(tool_results: Sequence[MCPToolResult], *, max_chars: int) -> str:
    if not tool_results:
        return "No tool evidence was used."
    blocks: list[str] = []
    used = 0
    for result in tool_results:
        if result.ok:
            text = _to_compact_json(result.data)
            prefix = f"TOOL {result.tool} OK:\n"
        else:
            text = result.error or "Unknown tool error"
            prefix = f"TOOL {result.tool} ERROR:\n"
        remaining = max_chars - used - len(prefix)
        if remaining <= 0:
            break
        text = text[:remaining]
        blocks.append(prefix + text)
        used += len(prefix) + len(text)
    return "\n\n".join(blocks)


def extract_references(tool_results: Sequence[MCPToolResult], *, limit: int) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(title: str | None, url: str | None, source: str) -> None:
        if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return
        if url in seen:
            return
        seen.add(url)
        refs.append({"title": (title or url).strip()[:180], "url": url.strip(), "source": source})

    def walk(obj: Any, source: str) -> None:
        if len(refs) >= limit:
            return
        if isinstance(obj, dict):
            url = obj.get("url") or obj.get("link") or obj.get("href")
            title = obj.get("title") or obj.get("name") or obj.get("source")
            add(title if isinstance(title, str) else None, url if isinstance(url, str) else None, source)
            for value in obj.values():
                walk(value, source)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, source)

    for result in tool_results:
        walk(result.data, result.tool)
        if len(refs) >= limit:
            break
    return refs[:limit]


def render_reference_instruction(references: Sequence[dict[str, str]]) -> str:
    if not references:
        return "No reference URLs were found in tool output."
    lines = ["Use these references when relevant:"]
    for idx, ref in enumerate(references, 1):
        lines.append(f"[{idx}] {ref['title']} - {ref['url']} ({ref['source']})")
    return "\n".join(lines)


def append_references(answer: str, references: Sequence[dict[str, str]]) -> str:
    answer = clean_model_output(answer)
    if not references:
        return answer
    if "references:" in answer.lower() or "sources:" in answer.lower():
        return answer
    lines = [answer.rstrip(), "", "References:"]
    for idx, ref in enumerate(references, 1):
        lines.append(f"{idx}. {ref['title']} — {ref['url']}")
    return "\n".join(lines)


def clean_model_output(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    text = THINKING_PROCESS_RE.sub("", text)
    return text.strip()


def _to_compact_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(data)
