from app.agents.base import StructuredAgent
from app.agents.prompts import WORKER_PROMPT, REVISER_PROMPT
from app.schemas.worker import WorkerResult

class WorkerAgent(StructuredAgent):
    async def execute(self, payload: dict) -> WorkerResult:
        return await self.invoke_json(system=WORKER_PROMPT, payload=payload, schema=WorkerResult)
    async def revise(self, payload: dict) -> WorkerResult:
        return await self.invoke_json(system=REVISER_PROMPT, payload=payload, schema=WorkerResult)
