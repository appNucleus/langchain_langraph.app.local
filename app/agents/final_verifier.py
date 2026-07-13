from app.agents.base import StructuredAgent
from app.agents.prompts import FINAL_VERIFIER_PROMPT
from app.schemas.finalization import FinalVerificationReport


class FinalVerifierAgent(StructuredAgent):
    async def verify_final(self, payload: dict) -> FinalVerificationReport:
        return await self.invoke_json(
            system=FINAL_VERIFIER_PROMPT,
            payload=payload,
            schema=FinalVerificationReport,
        )
