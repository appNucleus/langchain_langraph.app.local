from __future__ import annotations
import json
from typing import Any, AsyncIterator
from uuid import uuid4
from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.agents.worker import WorkerAgent
from app.agents.verifier import VerifierAgent
from app.agents.synthesizer import SynthesizerAgent
from app.graphs.state import AgentGraphState
from app.graphs.routes import after_verification, after_advance
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.planning import ExecutionPlan, PlanTask
from app.schemas.worker import WorkerResult
from app.schemas.verification import VerificationReport, VerificationIssue
from app.services.answer_quality import deterministic_output_issues
from app.services.context_builder import build_context
from app.services.evidence import evidence_from_metadata
from app.settings import Settings


def encode_sse(event: str, data: object) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

class _NoopDependency:
    async def health(self): return {'status': 'disabled'}
    async def health_check(self):
        class R: ok=True; data={'status':'disabled'}; error=None
        return R()
    async def aclose(self): return None

class _Selector:
    def __init__(self, settings: Settings): self.settings=settings

class ChatAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.planner = PlannerAgent(settings, getattr(settings, 'model_planner', None))
        self.worker = WorkerAgent(settings, getattr(settings, 'model_general', None))
        self.verifier = VerifierAgent(settings, getattr(settings, 'model_reasoning', None))
        self.synthesizer = SynthesizerAgent(settings, getattr(settings, 'model_synthesis', None))
        self.ollama = _NoopDependency()
        self.mcp = _NoopDependency()
        self.selector = _Selector(settings)
        self.graph = self._build_graph()

    async def start(self) -> None: return None
    async def aclose(self) -> None: return None
    async def load_inventory(self) -> dict[str, object]: return {'models': [], 'tools': []}

    def _build_graph(self):
        b = StateGraph(AgentGraphState)
        b.add_node('plan', self._plan)
        b.add_node('worker', self._worker)
        b.add_node('verify', self._verify)
        b.add_node('revise', self._revise)
        b.add_node('research', self._research)
        b.add_node('replan', self._replan)
        b.add_node('advance', self._advance)
        b.add_node('finalize', self._finalize)
        b.add_edge(START, 'plan')
        b.add_edge('plan', 'worker')
        b.add_edge('worker', 'verify')
        b.add_conditional_edges('verify', after_verification, {'advance':'advance','revise':'revise','research':'research','replan':'replan'})
        b.add_edge('revise', 'verify')
        b.add_edge('research', 'worker')
        b.add_edge('replan', 'worker')
        b.add_conditional_edges('advance', after_advance, {'worker':'worker','finalize':'finalize'})
        b.add_edge('finalize', END)
        return b.compile()

    async def _plan(self, state: AgentGraphState) -> AgentGraphState:
        if self.settings.llm_backend != 'ollama':
            plan = ExecutionPlan(goal=state['message'], tasks=[PlanTask(id='t1', objective=state['message'], completion_criteria=['Answer the request directly'])])
        else:
            plan = await self.planner.plan(state['message'])
        return {**state, 'plan': plan.model_dump(), 'task_index': 0, 'task_results': [], 'iterations': 0, 'research_rounds': 0, 'replans': 0}

    def _current_task(self, state: AgentGraphState) -> dict[str, Any]:
        return state['plan']['tasks'][state.get('task_index', 0)]

    async def _worker(self, state: AgentGraphState) -> AgentGraphState:
        task = self._current_task(state)
        evidence = evidence_from_metadata(state.get('metadata', {}))
        context = build_context(evidence, getattr(self.settings, 'phase2_max_context_chars', 16000))
        payload = {'user_request': state['message'], 'task': task, 'evidence': context, 'previous_verification': state.get('verification')}
        if self.settings.llm_backend != 'ollama':
            result = WorkerResult(answer=f"Echo mode is active. Message received: {state['message']}", confidence=0.5)
        else:
            result = await self.worker.execute(payload)
        return {**state, 'worker_result': result.model_dump(), 'evidence': [e.model_dump() for e in evidence], 'iterations': state.get('iterations',0)+1}

    async def _verify(self, state: AgentGraphState) -> AgentGraphState:
        max_iterations = getattr(self.settings, 'phase2_max_iterations', 4)
        if state.get('iterations', 0) >= max_iterations:
            report = VerificationReport(verdict='pass', task_complete=False, issues=[VerificationIssue(code='budget_exhausted', description='Maximum correction iterations reached.', severity='high')], required_actions=[], confidence=0.2)
            return {**state, 'verification': report.model_dump(), 'termination_reason': 'max_iterations'}
        issues = deterministic_output_issues(state['worker_result']['answer'])
        if self.settings.llm_backend != 'ollama':
            report = VerificationReport(verdict='pass', task_complete=True, issues=[VerificationIssue(code=i, description=i) for i in issues], confidence=0.5)
        else:
            report = await self.verifier.verify({'user_request': state['message'], 'task': self._current_task(state), 'worker_result': state['worker_result'], 'evidence': state.get('evidence', []), 'deterministic_issues': issues})
        return {**state, 'verification': report.model_dump()}

    async def _revise(self, state: AgentGraphState) -> AgentGraphState:
        payload = {'user_request': state['message'], 'task': self._current_task(state), 'worker_result': state['worker_result'], 'verification': state['verification'], 'evidence': state.get('evidence', [])}
        result = await self.worker.revise(payload) if self.settings.llm_backend == 'ollama' else WorkerResult.model_validate(state['worker_result'])
        return {**state, 'worker_result': result.model_dump(), 'iterations': state.get('iterations',0)+1}

    async def _research(self, state: AgentGraphState) -> AgentGraphState:
        rounds = state.get('research_rounds', 0) + 1
        max_rounds = getattr(self.settings, 'phase2_max_research_rounds', 2)
        if rounds > max_rounds:
            verification = dict(state['verification']); verification['verdict'] = 'revise'
            return {**state, 'verification': verification, 'research_rounds': rounds}
        metadata = dict(state.get('metadata', {}))
        metadata.setdefault('evidence', []).append({'id': f'research-{rounds}', 'source': 'verifier_request', 'content': 'Additional external research was requested but no MCP evidence provider was attached to this Phase 2 runtime.'})
        return {**state, 'metadata': metadata, 'research_rounds': rounds}

    async def _replan(self, state: AgentGraphState) -> AgentGraphState:
        replans = state.get('replans', 0) + 1
        if replans > getattr(self.settings, 'phase2_max_replans', 1):
            verification = dict(state['verification']); verification['verdict'] = 'revise'
            return {**state, 'verification': verification, 'replans': replans}
        plan = await self.planner.plan(state['message']) if self.settings.llm_backend == 'ollama' else ExecutionPlan.model_validate(state['plan'])
        return {**state, 'plan': plan.model_dump(), 'task_index': 0, 'replans': replans}

    async def _advance(self, state: AgentGraphState) -> AgentGraphState:
        results = list(state.get('task_results', [])); results.append({'task': self._current_task(state), 'worker_result': state['worker_result'], 'verification': state['verification']})
        return {**state, 'task_results': results, 'task_index': state.get('task_index',0)+1, 'verification': {}, 'worker_result': {}}

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        results = state.get('task_results', [])
        if self.settings.llm_backend == 'ollama' and len(results) > 1:
            response = await self.synthesizer.synthesize({'user_request': state['message'], 'verified_results': results})
        else:
            response = results[-1]['worker_result']['answer'] if results else state.get('worker_result', {}).get('answer', '')
        return {**state, 'response': response, 'backend': self.settings.llm_backend, 'model': getattr(self.settings, 'model_general', getattr(self.settings,'ollama_model',None))}

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        result = await self.graph.ainvoke({'message': request.message, 'system_prompt': request.system_prompt or self.settings.default_system_prompt, 'metadata': request.metadata})
        return ChatResponse.from_result(thread_id=request.thread_id, response=result['response'], backend=result['backend'], model=result.get('model'), metadata={**request.metadata, 'phase':'2', 'plan':result.get('plan'), 'verification':result.get('task_results'), 'iterations':result.get('iterations'), 'termination_reason':result.get('termination_reason')})

    async def astream_events(self, request: ChatRequest) -> AsyncIterator[dict[str, object]]:
        yield {'event':'status','data':{'stage':'started'}}
        response = await self.ainvoke(request)
        yield {'event':'final','data':response.model_dump()}
