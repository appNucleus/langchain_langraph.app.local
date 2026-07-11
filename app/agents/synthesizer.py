from app.agents.base import StructuredAgent
from app.agents.prompts import FINALIZER_PROMPT

class SynthesizerAgent(StructuredAgent):
    async def synthesize(self, payload: dict) -> str:
        return await self.invoke_text(system=FINALIZER_PROMPT, payload=payload)
