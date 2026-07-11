from app.agents.base import StructuredAgent
from app.agents.prompts import VERIFIER_PROMPT
from app.schemas.verification import VerificationReport

class VerifierAgent(StructuredAgent):
    async def verify(self, payload: dict) -> VerificationReport:
        return await self.invoke_json(system=VERIFIER_PROMPT, payload=payload, schema=VerificationReport)
