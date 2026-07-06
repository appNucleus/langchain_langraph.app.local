from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.settings import Settings


class ChatGraphState(TypedDict, total=False):
    message: str
    system_prompt: str
    response: str
    backend: str
    model: str | None
    metadata: dict[str, Any]


class ChatAgent:
    """Small LangGraph wrapper that is easy to extend later.

    Current graph:
        START -> assistant -> END

    Later extension points:
        START -> classify_intent -> retrieve_context -> call_mcp_tools -> assistant -> validate -> END
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ChatGraphState)
        builder.add_node("assistant", self._assistant_node)
        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)
        return builder.compile()

    async def _assistant_node(self, state: ChatGraphState) -> ChatGraphState:
        message = state["message"]
        system_prompt = state.get("system_prompt") or self.settings.default_system_prompt

        if self.settings.llm_backend == "ollama":
            response = await self._call_ollama(message=message, system_prompt=system_prompt)
            return {
                **state,
                "response": response,
                "backend": "ollama",
                "model": self.settings.ollama_model,
                "metadata": {
                    **state.get("metadata", {}),
                    "ollama_base_url": self.settings.ollama_base_url,
                },
            }

        return {
            **state,
            "response": (
                "Echo mode is active. Your FastAPI + LangGraph service is running. "
                f"Message received: {message}"
            ),
            "backend": "echo",
            "model": None,
            "metadata": {
                **state.get("metadata", {}),
                "next_step": "Set LLM_BACKEND=ollama after your Ollama server/model is ready.",
            },
        }

    async def _call_ollama(self, *, message: str, system_prompt: str) -> str:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            temperature=self.settings.ollama_temperature,
            timeout=self.settings.ollama_timeout_seconds,
        )
        result = await llm.ainvoke(
            [
                ("system", system_prompt),
                ("human", message),
            ]
        )
        content = getattr(result, "content", result)
        if isinstance(content, str):
            return content
        return str(content)
