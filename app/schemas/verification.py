from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["pass", "revise", "research", "replan", "terminate"]


class VerificationIssue(BaseModel):
    code: str
    description: str
    severity: Literal["low", "medium", "high"] = "medium"
    claim_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    verdict: Verdict
    task_complete: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
