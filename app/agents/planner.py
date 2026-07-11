from app.agents.base import StructuredAgent
from app.agents.prompts import PLANNER_PROMPT
from app.schemas.planning import ExecutionPlan

class PlannerAgent(StructuredAgent):
    async def plan(self, message: str) -> ExecutionPlan:
        return await self.invoke_json(system=PLANNER_PROMPT, payload={'request': message}, schema=ExecutionPlan)
