from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.llm.ollama import OllamaClient
from app.logging_config import log_kv
from app.mcp.client import MCPClient, MCPToolResult
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.formatting import (
    PromptBuilder,
    append_references,
    clean_model_output,
    extract_references,
)
from app.services.memory import ConversationTurn, InMemoryConversationStore
from app.services.query_rewriter import QueryRewriter
from app.services.routing import (
    QueryPlan,
    extract_location,
    extract_road,
    extract_ticker,
    extract_urls,
)
from app.services.routing import ModelRouter
from app.settings import Settings

logger = logging.getLogger(__name__)


class ChatGraphState(TypedDict, total=False):
    thread_id: str
    message: str
    system_prompt: str
    metadata: dict[str, Any]
    history: list[ConversationTurn]
    plan: QueryPlan
    rewritten_query: str | None
    tool_results: list[MCPToolResult]
    references: list[dict[str, str]]
    response: str
    backend: str
    model: str | None


class ChatAgent:
    """LangGraph-backed chat orchestrator.

    Graph path for non-streaming calls:
        START -> prepare -> plan -> rewrite -> call_tools -> answer -> END

    Streaming calls use the same node methods but stream the final Ollama call
    token-by-token as SSE events.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        ollama_client: OllamaClient | None = None,
        mcp_client: MCPClient | None = None,
        memory: InMemoryConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.ollama = ollama_client or OllamaClient(settings)
        self.mcp = mcp_client or MCPClient(settings)
        self.memory = memory or InMemoryConversationStore(settings)
        self.router = ModelRouter(settings)
        self.rewriter = QueryRewriter(settings)
        self.prompt_builder = PromptBuilder(settings, self.memory)
        self.graph = self._build_graph()
        log_kv(
            logger,
            logging.INFO,
            "agent_initialized",
            backend=settings.llm_backend,
            mcp_enabled=settings.mcp_enabled,
            model_general=settings.model_general,
            model_search=settings.model_search,
        )

    def _build_graph(self):
        builder = StateGraph(ChatGraphState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("call_tools", self._tools_node)
        builder.add_node("answer", self._answer_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "plan")
        builder.add_edge("plan", "rewrite")
        builder.add_edge("rewrite", "call_tools")
        builder.add_edge("call_tools", "answer")
        builder.add_edge("answer", END)
        return builder.compile()

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        thread_id = request.thread_id or str(uuid4())
        log_kv(logger, logging.INFO, "ainvoke_start", thread_id=thread_id, message_chars=len(request.message))
        initial: ChatGraphState = {
            "thread_id": thread_id,
            "message": request.message,
            "system_prompt": self._system_prompt_or_default(request.system_prompt),
            "metadata": dict(request.metadata or {}),
        }
        result = await self.graph.ainvoke(initial)
        response_text = result.get("response", "")
        log_kv(
            logger,
            logging.INFO,
            "ainvoke_done",
            thread_id=thread_id,
            backend=result.get("backend", self.settings.llm_backend),
            model=result.get("model"),
            response_chars=len(response_text),
        )
        await self.memory.add_pair(thread_id, user=request.message, assistant=response_text)
        return ChatResponse.from_result(
            thread_id=thread_id,
            response=response_text,
            backend=result.get("backend", self.settings.llm_backend),
            model=result.get("model"),
            metadata=self._response_metadata(result),
        )

    async def astream_events(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        thread_id = request.thread_id or str(uuid4())
        log_kv(logger, logging.INFO, "astream_start", thread_id=thread_id, message_chars=len(request.message))
        state: ChatGraphState = {
            "thread_id": thread_id,
            "message": request.message,
            "system_prompt": self._system_prompt_or_default(request.system_prompt),
            "metadata": dict(request.metadata or {}),
        }
        yield {"event": "status", "data": {"status": "started", "thread_id": thread_id}}

        state.update(await self._prepare_node(state))
        state.update(await self._plan_node(state))
        plan = state["plan"]
        model = self.settings.model_for_key(plan.model_key)
        yield {
            "event": "plan",
            "data": {
                "intent": plan.intent,
                "tools": plan.tools,
                "model": model,
                "reason": plan.reason,
            },
        }

        state.update(await self._rewrite_node(state))
        if state.get("rewritten_query"):
            yield {"event": "query", "data": {"query": state["rewritten_query"]}}

        if plan.tools and self.settings.mcp_enabled:
            for tool in plan.tools[: self.settings.max_tool_calls]:
                yield {"event": "tool_start", "data": {"tool": tool}}
            state.update(await self._tools_node(state))
            for result in state.get("tool_results", []):
                yield {
                    "event": "tool_result",
                    "data": {"tool": result.tool, "ok": result.ok, "error": result.error},
                }
        else:
            state.update({"tool_results": [], "references": []})

        if self.settings.llm_backend == "echo":
            response = self._echo_response(state)
            for chunk in _chunk_text(response):
                yield {"event": "token", "data": {"delta": chunk}}
                await asyncio.sleep(0)
            state.update({"response": response, "backend": "echo", "model": None})
        else:
            messages = self._messages_for_state(state)
            response_parts: list[str] = []
            try:
                async for delta in self.ollama.stream_chat(
                    model=model,
                    messages=messages,
                    temperature=self.settings.ollama_temperature,
                    num_predict=self.settings.ollama_stream_num_predict,
                ):
                    # OllamaClient already suppresses message.thinking. Do not strip
                    # every token here, because stripping breaks spaces between chunks.
                    clean_delta = delta
                    if not clean_delta:
                        continue
                    response_parts.append(clean_delta)
                    yield {"event": "token", "data": {"delta": clean_delta}}
            except Exception as exc:  # noqa: BLE001 - report as SSE error and fallback.
                yield {"event": "error", "data": {"message": str(exc)}}
            response = clean_model_output("".join(response_parts))
            if not response:
                response = "I could not get answer content from the local Ollama model."
            response = append_references(response, state.get("references", []))
            state.update({"response": response, "backend": "ollama", "model": model})

        await self.memory.add_pair(thread_id, user=request.message, assistant=state.get("response", ""))
        yield {
            "event": "done",
            "data": {
                "thread_id": thread_id,
                "backend": state.get("backend", self.settings.llm_backend),
                "model": state.get("model"),
                "metadata": self._response_metadata(state),
            },
        }

    async def _prepare_node(self, state: ChatGraphState) -> ChatGraphState:
        history = await self.memory.get(state["thread_id"])
        return {**state, "history": history}

    async def _plan_node(self, state: ChatGraphState) -> ChatGraphState:
        plan = self.router.plan(state["message"], state.get("metadata", {}))
        model = self.settings.model_for_key(plan.model_key)
        log_kv(
            logger,
            logging.INFO,
            "route_plan",
            thread_id=state.get("thread_id"),
            intent=plan.intent,
            tools=",".join(plan.tools),
            model=model,
            reason=plan.reason,
        )
        return {**state, "plan": plan, "model": model}

    async def _rewrite_node(self, state: ChatGraphState) -> ChatGraphState:
        plan = state["plan"]
        rewritten = None
        if plan.needs_query_rewrite:
            rewritten = self.rewriter.rewrite(
                state["message"],
                plan,
                metadata=state.get("metadata", {}),
            )
            log_kv(logger, logging.INFO, "query_rewritten", thread_id=state.get("thread_id"), query=rewritten)
        return {**state, "rewritten_query": rewritten}

    async def _tools_node(self, state: ChatGraphState) -> ChatGraphState:
        plan = state["plan"]
        if not self.settings.mcp_enabled or not plan.tools:
            return {**state, "tool_results": [], "references": []}

        if self.settings.llm_backend == "echo":
            log_kv(
                logger,
                logging.WARNING,
                "mcp_skipped_echo_backend",
                thread_id=state.get("thread_id"),
                requested_tools=",".join(plan.tools),
            )
            return {**state, "tool_results": [], "references": []}

        requested_tools = plan.tools[: self.settings.max_tool_calls]
        prepared: list[tuple[str, dict[str, Any] | None]] = [(tool, self._tool_args(tool, state)) for tool in requested_tools]

        async def run_tool(tool: str, args: dict[str, Any] | None) -> MCPToolResult:
            if args is None:
                log_kv(logger, logging.WARNING, "mcp_tool_missing_args", thread_id=state.get("thread_id"), tool=tool)
                return MCPToolResult(tool=tool, ok=False, data=None, error="Missing required tool arguments.")
            log_kv(
                logger,
                logging.INFO,
                "mcp_tool_start",
                thread_id=state.get("thread_id"),
                tool=tool,
                arg_keys=",".join(sorted(args.keys())),
            )
            result = await self.mcp.call_tool(tool, args)
            log_kv(
                logger,
                logging.INFO if result.ok else logging.ERROR,
                "mcp_tool_done",
                thread_id=state.get("thread_id"),
                tool=tool,
                ok=result.ok,
                error=result.error,
            )
            return result

        results = list(await asyncio.gather(*(run_tool(tool, args) for tool, args in prepared)))
        references = extract_references(results, limit=self.settings.max_references)
        log_kv(logger, logging.INFO, "mcp_tools_complete", thread_id=state.get("thread_id"), refs=len(references))
        return {**state, "tool_results": results, "references": references}

    async def _answer_node(self, state: ChatGraphState) -> ChatGraphState:
        if self.settings.llm_backend == "echo":
            log_kv(logger, logging.WARNING, "echo_backend_response", thread_id=state.get("thread_id"))
            return {**state, "response": self._echo_response(state), "backend": "echo", "model": None}

        model = state.get("model") or self.settings.model_for_key(state["plan"].model_key)
        messages = self._messages_for_state(state)
        log_kv(logger, logging.INFO, "ollama_answer_start", thread_id=state.get("thread_id"), model=model, messages=len(messages))
        result = await self.ollama.chat(
            model=model,
            messages=messages,
            temperature=self.settings.ollama_temperature,
            num_predict=self.settings.ollama_num_predict,
        )
        response = append_references(clean_model_output(result.content), state.get("references", []))
        if not response:
            response = "I could not get answer content from the local Ollama model."
        log_kv(logger, logging.INFO, "ollama_answer_done", thread_id=state.get("thread_id"), model=result.model, response_chars=len(response))
        return {**state, "response": response, "backend": "ollama", "model": result.model}

    def _messages_for_state(self, state: ChatGraphState) -> list[dict[str, str]]:
        return self.prompt_builder.build_messages(
            user_message=state["message"],
            system_prompt=state.get("system_prompt") or self.settings.default_system_prompt,
            history=state.get("history", []),
            plan=state["plan"],
            rewritten_query=state.get("rewritten_query"),
            tool_results=state.get("tool_results", []),
            references=state.get("references", []),
        )

    def _tool_args(self, tool: str, state: ChatGraphState) -> dict[str, Any] | None:
        message = state["message"]
        metadata = state.get("metadata", {})
        rewritten = state.get("rewritten_query") or message
        urls = extract_urls(message)

        args: dict[str, Any] | None
        if tool == "health_check":
            args = {}
        elif tool == "web_search":
            args = {
                "query": rewritten,
                "max_results": int(metadata.get("max_results", self.settings.default_search_results)),
                "language": metadata.get("language"),
                "categories": metadata.get("categories"),
                "time_range": metadata.get("time_range"),
            }
        elif tool == "web_search_and_scrape":
            args = {
                "query": rewritten,
                "pages": int(metadata.get("pages", self.settings.default_scrape_pages)),
                "max_chars_per_page": metadata.get("max_chars_per_page"),
                "include_images": bool(metadata.get("include_images", True)),
                "render_js": metadata.get("render_js"),
                "language": metadata.get("language"),
                "categories": metadata.get("categories"),
                "time_range": metadata.get("time_range"),
                "prefer_official": bool(metadata.get("prefer_official", True)),
            }
        elif tool == "scrape_url":
            args = (
                {
                    "url": urls[0],
                    "include_images": bool(metadata.get("include_images", True)),
                    "max_chars": metadata.get("max_chars"),
                    "render_js": metadata.get("render_js"),
                    "query_context": message,
                }
                if urls
                else None
            )
        elif tool == "extract_image_urls":
            args = (
                {
                    "url": urls[0],
                    "max_images": int(metadata.get("max_images", 20)),
                    "render_js": metadata.get("render_js"),
                    "query_context": message,
                }
                if urls
                else None
            )
        elif tool == "weather_lookup":
            args = {
                "location": extract_location(message, metadata),
                "forecast_days": int(metadata.get("forecast_days", self.settings.default_forecast_days)),
                "include_hourly": bool(metadata.get("include_hourly", False)),
                "include_nws": bool(metadata.get("include_nws", True)),
            }
        elif tool == "stock_quote":
            symbol = extract_ticker(message) or str(metadata.get("symbol", "")).upper()
            args = {"symbol": symbol} if symbol else None
        elif tool == "stock_news":
            symbol = extract_ticker(message) or str(metadata.get("symbol", "")).upper()
            args = (
                {
                    "symbol": symbol,
                    "max_items": int(metadata.get("max_items", self.settings.default_search_results)),
                    "lookback_days": int(metadata.get("lookback_days", self.settings.default_news_lookback_days)),
                }
                if symbol
                else None
            )
        elif tool == "explain_stock_move":
            symbol = extract_ticker(message) or str(metadata.get("symbol", "")).upper()
            args = (
                {
                    "symbol": symbol,
                    "lookback_days": int(metadata.get("lookback_days", self.settings.default_news_lookback_days)),
                    "max_news": int(metadata.get("max_news", self.settings.default_search_results)),
                }
                if symbol
                else None
            )
        elif tool == "news_search":
            args = {
                "query": rewritten,
                "max_items": int(metadata.get("max_items", self.settings.default_search_results)),
                "lookback_days": int(metadata.get("lookback_days", self.settings.default_news_lookback_days)),
            }
        elif tool == "road_condition_search":
            args = {
                "road": extract_road(message) or str(metadata.get("road", "road")),
                "location": extract_location(message, metadata),
                "when": str(metadata.get("when", "now")),
                "pages": int(metadata.get("pages", 2)),
            }
        elif tool == "mail_search":
            args = {"query": rewritten, "max_results": int(metadata.get("max_results", self.settings.default_search_results))}
        elif tool == "mail_read":
            message_id = metadata.get("message_id") or metadata.get("mail_message_id")
            args = {"message_id": str(message_id)} if message_id else None
        elif tool == "mail_create_draft":
            to = metadata.get("to")
            subject = metadata.get("subject", "Draft from local assistant")
            body = metadata.get("body") or message
            args = {"to": str(to), "subject": str(subject), "body": str(body), "cc": metadata.get("cc")} if to else None
        elif tool == "mail_send_draft":
            # Intentionally guarded: this only works when the caller supplies both
            # draft_id and confirmation_token. Natural language alone is not enough.
            draft_id = metadata.get("draft_id")
            confirmation_token = metadata.get("confirmation_token")
            args = {"draft_id": str(draft_id), "confirmation_token": str(confirmation_token)} if draft_id and confirmation_token else None
        else:
            args = None

        args = self._apply_tool_arg_overrides(tool, args, metadata)
        return _drop_none(args) if args is not None else None

    @staticmethod
    def _apply_tool_arg_overrides(tool: str, args: dict[str, Any] | None, metadata: dict[str, Any]) -> dict[str, Any] | None:
        overrides = metadata.get("tool_args") or metadata.get("mcp_tool_args") or {}
        if not isinstance(overrides, dict):
            return args
        tool_overrides = overrides.get(tool)
        if isinstance(tool_overrides, dict):
            return {**(args or {}), **tool_overrides}
        return args


    def _system_prompt_or_default(self, system_prompt: str | None) -> str:
        # Swagger/OpenAPI examples often send the literal placeholder "string".
        # Treat that as no custom system prompt so the assistant keeps its real instructions.
        if not system_prompt or system_prompt.strip().lower() == "string":
            return self.settings.default_system_prompt
        return system_prompt

    def _echo_response(self, state: ChatGraphState) -> str:
        plan = state.get("plan")
        return (
            "Echo mode is active. FastAPI + LangGraph orchestration is running. "
            f"Intent: {plan.intent if plan else 'unknown'}. "
            f"Message received: {state['message']}"
        )

    def _response_metadata(self, state: ChatGraphState) -> dict[str, Any]:
        plan = state.get("plan")
        results = state.get("tool_results", [])
        return {
            **state.get("metadata", {}),
            "intent": plan.intent if plan else None,
            "tools_requested": plan.tools if plan else [],
            "tools_used": [item.tool for item in results if item.ok],
            "tool_errors": [
                {"tool": item.tool, "error": item.error}
                for item in results
                if not item.ok
            ],
            "rewritten_query": state.get("rewritten_query"),
            "references": state.get("references", []),
        }


def encode_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 80) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}
