from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.llm.ollama import OllamaClient
from app.logging_config import log_kv
from app.mcp.client import MCPClient, MCPToolResult
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.answer_quality import validate_answer
from app.services.decomposition import QueryDecomposer, QueryTask
from app.services.formatting import (
    PromptBuilder,
    append_references,
    clean_model_output,
    extract_references,
)
from app.services.inventory import InventoryService, ModelSelector, RuntimeInventory
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
    inventory: RuntimeInventory
    plan: QueryPlan
    tasks: list[QueryTask]
    rewritten_query: str | None
    tool_results: list[MCPToolResult]
    task_tool_results: dict[str, list[MCPToolResult]]
    task_answers: list[dict[str, Any]]
    references: list[dict[str, str]]
    response: str
    backend: str
    model: str | None
    models_used: list[str]


class ChatAgent:
    """LangGraph-backed chat orchestrator.

    Non-streaming path:
        START -> prepare -> inventory -> plan -> rewrite -> call_tools -> answer -> END

    The graph now plans against live Ollama/MCP inventory, can split compound
    requests into simple tasks, answers each task with the best available model
    and tools, then synthesizes a clean final answer.
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
        self.selector = ModelSelector(settings)
        self.inventory_service = InventoryService(settings, self.ollama, self.mcp)
        self.decomposer = QueryDecomposer(settings, self.router, self.selector)
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

    async def start(self) -> None:
        for dependency in (self.ollama, self.mcp):
            start = getattr(dependency, "start", None)
            if callable(start):
                await start()

    async def aclose(self) -> None:
        for dependency in (self.mcp, self.ollama):
            close = getattr(dependency, "aclose", None)
            if callable(close):
                await close()

    def _build_graph(self):
        builder = StateGraph(ChatGraphState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("inventory", self._inventory_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("call_tools", self._tools_node)
        builder.add_node("answer", self._answer_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "inventory")
        builder.add_edge("inventory", "plan")
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
        state.update(await self._inventory_node(state))

        state.update(await self._plan_node(state))
        tasks = state.get("tasks", [])
        plan = state["plan"]
        yield {
            "event": "plan",
            "data": {
                "intent": plan.intent,
                "tools": plan.tools,
                "model": state.get("model"),
                "reason": plan.reason,
                "tasks": [task.as_metadata() for task in tasks],
            },
        }

        state.update(await self._rewrite_node(state))
        if state.get("rewritten_query"):
            yield {"event": "query", "data": {"query": state["rewritten_query"]}}

        all_tools = [tool for task in state.get("tasks", []) for tool in task.tools]
        if all_tools and self.settings.mcp_enabled:
            for tool in all_tools[: self.settings.max_tool_calls * max(1, len(tasks))]:
                yield {"event": "tool_start", "data": {"tool": tool}}
            state.update(await self._tools_node(state))
            for result in state.get("tool_results", []):
                yield {
                    "event": "tool_result",
                    "data": {"tool": result.tool, "ok": result.ok, "error": result.error},
                }
        else:
            state.update({"tool_results": [], "task_tool_results": {}, "references": []})

        state.update(await self._answer_node(state))
        for chunk in _chunk_text(state.get("response", "")):
            yield {"event": "token", "data": {"delta": chunk}}
            await asyncio.sleep(0)

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

    async def load_inventory(self) -> RuntimeInventory:
        return await self.inventory_service.load()

    async def _prepare_node(self, state: ChatGraphState) -> ChatGraphState:
        history = await self.memory.get(state["thread_id"])
        return {**state, "history": history}

    async def _inventory_node(self, state: ChatGraphState) -> ChatGraphState:
        inventory = await self.load_inventory()
        log_kv(
            logger,
            logging.INFO,
            "inventory_loaded",
            thread_id=state.get("thread_id"),
            models=len(inventory.model_names),
            tools=len(inventory.tool_names),
            errors=",".join(inventory.errors.keys()),
        )
        return {**state, "inventory": inventory}

    async def _plan_node(self, state: ChatGraphState) -> ChatGraphState:
        inventory = state.get("inventory") or RuntimeInventory()
        metadata = state.get("metadata", {})
        queries = await self.decomposer.decompose(
            message=state["message"],
            metadata=metadata,
            inventory=inventory,
            ollama=self.ollama,
        )
        tasks = self.decomposer.build_tasks(queries=queries, metadata=metadata, inventory=inventory)
        if not tasks:
            fallback_plan = self.router.plan(state["message"], metadata)
            model = self.selector.resolve(fallback_plan.model_key, inventory)
            tasks = [QueryTask(id="q1", query=state["message"], plan=fallback_plan, model=model, tools=fallback_plan.tools)]
        plan = tasks[0].plan
        model = tasks[0].model if len(tasks) == 1 else self.selector.resolve("synthesis", inventory)
        log_kv(
            logger,
            logging.INFO,
            "route_plan",
            thread_id=state.get("thread_id"),
            intent=plan.intent,
            tools=",".join(plan.tools),
            model=model,
            task_count=len(tasks),
            reason=plan.reason,
        )
        return {**state, "plan": plan, "tasks": tasks, "model": model}

    async def _rewrite_node(self, state: ChatGraphState) -> ChatGraphState:
        rewritten_first: str | None = None
        rewritten_tasks: list[QueryTask] = []
        for task in state.get("tasks", []):
            rewritten = None
            if task.plan.needs_query_rewrite:
                rewritten = self.rewriter.rewrite(task.query, task.plan, metadata=state.get("metadata", {}))
                log_kv(logger, logging.INFO, "query_rewritten", thread_id=state.get("thread_id"), task_id=task.id, query=rewritten)
            rewritten_tasks.append(
                QueryTask(
                    id=task.id,
                    query=task.query,
                    plan=task.plan,
                    model=task.model,
                    tools=task.tools,
                    rewritten_query=rewritten,
                )
            )
            if rewritten_first is None:
                rewritten_first = rewritten
        return {**state, "tasks": rewritten_tasks, "rewritten_query": rewritten_first}

    async def _tools_node(self, state: ChatGraphState) -> ChatGraphState:
        tasks = state.get("tasks", [])
        if not self.settings.mcp_enabled or not tasks:
            return {**state, "tool_results": [], "task_tool_results": {}, "references": []}

        if self.settings.llm_backend == "echo":
            requested = [tool for task in tasks for tool in task.tools]
            log_kv(
                logger,
                logging.WARNING,
                "mcp_skipped_echo_backend",
                thread_id=state.get("thread_id"),
                requested_tools=",".join(requested),
            )
            return {**state, "tool_results": [], "task_tool_results": {}, "references": []}

        async def run_tool(task: QueryTask, tool: str) -> tuple[str, MCPToolResult]:
            args = self._tool_args(tool, state, task=task)
            if args is None:
                log_kv(logger, logging.WARNING, "mcp_tool_missing_args", thread_id=state.get("thread_id"), task_id=task.id, tool=tool)
                return task.id, MCPToolResult(tool=tool, ok=False, data=None, error="Missing required tool arguments.")
            log_kv(
                logger,
                logging.INFO,
                "mcp_tool_start",
                thread_id=state.get("thread_id"),
                task_id=task.id,
                tool=tool,
                arg_keys=",".join(sorted(args.keys())),
            )
            result = await self.mcp.call_tool(tool, args)
            log_kv(
                logger,
                logging.INFO if result.ok else logging.ERROR,
                "mcp_tool_done",
                thread_id=state.get("thread_id"),
                task_id=task.id,
                tool=tool,
                ok=result.ok,
                error=result.error,
            )
            return task.id, result

        calls = [run_tool(task, tool) for task in tasks for tool in task.tools[: self.settings.max_tool_calls]]
        pairs = list(await asyncio.gather(*calls)) if calls else []
        task_results: dict[str, list[MCPToolResult]] = {task.id: [] for task in tasks}
        results: list[MCPToolResult] = []
        for task_id, result in pairs:
            task_results.setdefault(task_id, []).append(result)
            results.append(result)
        references = extract_references(results, limit=self.settings.max_references)
        log_kv(logger, logging.INFO, "mcp_tools_complete", thread_id=state.get("thread_id"), refs=len(references), results=len(results))
        return {**state, "tool_results": results, "task_tool_results": task_results, "references": references}

    async def _answer_node(self, state: ChatGraphState) -> ChatGraphState:
        if self.settings.llm_backend == "echo":
            log_kv(logger, logging.WARNING, "echo_backend_response", thread_id=state.get("thread_id"))
            return {**state, "response": self._echo_response(state), "backend": "echo", "model": None, "models_used": []}

        tasks = state.get("tasks", [])
        if len(tasks) <= 1:
            task = tasks[0] if tasks else QueryTask(
                id="q1",
                query=state["message"],
                plan=state["plan"],
                model=state.get("model") or self.selector.resolve(state["plan"].model_key, state.get("inventory")),
                tools=state["plan"].tools,
                rewritten_query=state.get("rewritten_query"),
            )
            task_answer, models_used = await self._answer_one_task(state, task, state.get("tool_results", []))
            response = append_references(
                task_answer["answer"],
                state.get("references", []),
                limit=self.settings.answer_reference_limit,
            )
            if not response:
                response = "I could not get answer content from the local Ollama model."
            model = models_used[-1] if models_used else task.model
            log_kv(logger, logging.INFO, "ollama_answer_done", thread_id=state.get("thread_id"), model=model, response_chars=len(response))
            return {
                **state,
                "task_answers": [task_answer],
                "response": response,
                "backend": "ollama",
                "model": model,
                "models_used": models_used,
            }

        task_answers: list[dict[str, Any]] = []
        models_used: list[str] = []
        task_tool_results = state.get("task_tool_results", {})
        for task in tasks:
            answer_payload, used_for_task = await self._answer_one_task(state, task, task_tool_results.get(task.id, []))
            models_used.extend(used_for_task)
            task_answers.append(answer_payload)

        synthesis_model = self.selector.resolve("synthesis", state.get("inventory"))
        synthesis_messages = self.prompt_builder.build_synthesis_messages(
            original_message=state["message"],
            system_prompt=state.get("system_prompt") or self.settings.default_system_prompt,
            history=state.get("history", []),
            task_answers=task_answers,
            references=state.get("references", []),
        )
        log_kv(logger, logging.INFO, "ollama_synthesis_start", thread_id=state.get("thread_id"), model=synthesis_model, task_count=len(task_answers))
        result = await self.ollama.chat(
            model=synthesis_model,
            messages=synthesis_messages,
            temperature=self.settings.ollama_temperature,
            num_predict=self.settings.ollama_num_predict,
        )
        response = clean_model_output(result.content)
        validation = validate_answer(
            response,
            query=state["message"],
            min_chars=self.settings.min_answer_chars,
            required_references=bool(state.get("references")),
            references_count=len(state.get("references", [])),
        )
        models_used.append(result.model)
        if not validation.ok:
            heavy_model = self.selector.resolve("heavy", state.get("inventory"))
            if heavy_model != result.model:
                retry_messages = [
                    *synthesis_messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous synthesis was not acceptable because: "
                            f"{validation.reasons}. Rewrite a complete, useful final answer."
                        ),
                    },
                ]
                retry = await self.ollama.chat(
                    model=heavy_model,
                    messages=retry_messages,
                    temperature=self.settings.ollama_temperature,
                    num_predict=self.settings.ollama_num_predict,
                )
                retry_response = clean_model_output(retry.content)
                retry_validation = validate_answer(
                    retry_response,
                    query=state["message"],
                    min_chars=self.settings.min_answer_chars,
                    required_references=bool(state.get("references")),
                    references_count=len(state.get("references", [])),
                )
                models_used.append(retry.model)
                if retry_validation.ok or len(retry_response) > len(response):
                    response = retry_response
                    validation = retry_validation
                    result = retry

        response = append_references(
            response,
            state.get("references", []),
            limit=self.settings.answer_reference_limit,
        )
        if not response:
            response = "I could not combine the answer content from the local Ollama model."
        return {
            **state,
            "task_answers": task_answers,
            "response": response,
            "backend": "ollama",
            "model": result.model,
            "models_used": models_used,
            "synthesis_validation": validation.as_dict(),
        }

    async def _answer_one_task(
        self,
        state: ChatGraphState,
        task: QueryTask,
        tool_results: Sequence[MCPToolResult],
    ) -> tuple[dict[str, Any], list[str]]:
        models_used: list[str] = []
        inventory = state.get("inventory")
        references_count = len(extract_references(tool_results, limit=self.settings.max_references))
        required_references = bool(task.tools) and task.plan.intent in {"web_research", "news", "stock", "sports_or_match_results", "url_scrape"}
        tried: list[str] = []
        last_answer = ""
        last_validation = None

        for attempt, model in enumerate(self._model_retry_sequence(task, inventory), 1):
            if model in tried:
                continue
            tried.append(model)
            task_state: ChatGraphState = {
                **state,
                "message": task.query,
                "plan": task.plan,
                "rewritten_query": task.rewritten_query,
                "tool_results": list(tool_results),
            }
            messages = self._messages_for_state(task_state)
            if attempt > 1 and last_validation is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous answer failed quality checks: "
                            f"{last_validation.reasons}. Give a complete answer that directly addresses the query."
                        ),
                    }
                )
            log_kv(logger, logging.INFO, "ollama_task_answer_start", thread_id=state.get("thread_id"), task_id=task.id, model=model, attempt=attempt)
            result = await self.ollama.chat(
                model=model,
                messages=messages,
                temperature=self.settings.ollama_temperature,
                num_predict=self.settings.ollama_num_predict,
            )
            answer = clean_model_output(result.content)
            validation = validate_answer(
                answer,
                query=task.query,
                min_chars=self.settings.min_answer_chars,
                required_references=required_references,
                references_count=references_count,
            )
            models_used.append(result.model)
            last_answer = answer
            last_validation = validation
            if validation.ok or attempt >= self.settings.max_answer_retries + 1:
                return {
                    "id": task.id,
                    "query": task.query,
                    "intent": task.plan.intent,
                    "model": result.model,
                    "model_key": task.plan.model_key,
                    "tools": task.tools,
                    "answer": answer or "No answer content returned.",
                    "validation": validation.as_dict(),
                    "retry_count": attempt - 1,
                }, models_used

        fallback_validation = last_validation or validate_answer(last_answer, query=task.query, min_chars=self.settings.min_answer_chars)
        return {
            "id": task.id,
            "query": task.query,
            "intent": task.plan.intent,
            "model": tried[-1] if tried else task.model,
            "model_key": task.plan.model_key,
            "tools": task.tools,
            "answer": last_answer or "No answer content returned.",
            "validation": fallback_validation.as_dict(),
            "retry_count": max(0, len(tried) - 1),
        }, models_used

    def _model_retry_sequence(self, task: QueryTask, inventory: RuntimeInventory | None) -> list[str]:
        sequence = [task.model]
        retry_keys = {
            "simple": ["general", "reasoning", "fallback"],
            "general": ["reasoning", "synthesis", "fallback"],
            "search": ["reasoning", "synthesis", "fallback"],
            "reasoning": ["fast_reasoning", "heavy", "fallback"],
            "fast_reasoning": ["reasoning", "heavy", "fallback"],
            "writer": ["synthesis", "heavy", "fallback"],
            "classifier": ["general", "fallback"],
            "vision": ["general", "fallback"],
            "fallback": ["general", "synthesis"],
        }.get(task.plan.model_key, ["reasoning", "synthesis", "fallback"])
        for key in retry_keys:
            model = self.selector.resolve(key, inventory)
            if model not in sequence:
                sequence.append(model)
        return sequence[: self.settings.max_answer_retries + 1]

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

    def _tool_args(self, tool: str, state: ChatGraphState, *, task: QueryTask | None = None) -> dict[str, Any] | None:
        message = task.query if task else state["message"]
        metadata = state.get("metadata", {})
        rewritten = (task.rewritten_query if task else state.get("rewritten_query")) or message
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
        if not system_prompt or system_prompt.strip().lower() == "string":
            return self.settings.default_system_prompt
        return system_prompt

    def _echo_response(self, state: ChatGraphState) -> str:
        plan = state.get("plan")
        tasks = state.get("tasks", [])
        return (
            "Echo mode is active. FastAPI + LangGraph orchestration is running. "
            f"Intent: {plan.intent if plan else 'unknown'}. "
            f"Tasks: {len(tasks)}. "
            f"Message received: {state['message']}"
        )

    def _response_metadata(self, state: ChatGraphState) -> dict[str, Any]:
        plan = state.get("plan")
        results = state.get("tool_results", [])
        inventory = state.get("inventory")
        tasks = state.get("tasks", [])
        tool_decisions = [
            {
                "subquery_id": task.id,
                "tool": tool,
                "reason": task.plan.reason,
            }
            for task in tasks
            for tool in task.tools
        ]
        metadata = {
            **state.get("metadata", {}),
            "intent": plan.intent if plan else None,
            "tools_requested": plan.tools if plan else [],
            "tools_used": [item.tool for item in results if item.ok],
            "tool_errors": [
                {"tool": item.tool, "error": item.error}
                for item in results
                if not item.ok
            ],
            "tool_decisions": tool_decisions,
            "rewritten_query": state.get("rewritten_query"),
            "subqueries": [task.as_metadata() for task in tasks],
            "task_answers": state.get("task_answers", []),
            "models_used": state.get("models_used", []),
            "references": state.get("references", []),
        }
        if inventory:
            metadata["inventory_summary"] = {
                "model_count": len(inventory.model_names),
                "tool_count": len(inventory.tool_names),
                "errors": inventory.errors,
            }
        if state.get("synthesis_validation"):
            metadata["synthesis_validation"] = state["synthesis_validation"]
        return metadata


def encode_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 80) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}
