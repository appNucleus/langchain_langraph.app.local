from __future__ import annotations
from pydantic import BaseModel, Field

class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

class WorkerResult(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
