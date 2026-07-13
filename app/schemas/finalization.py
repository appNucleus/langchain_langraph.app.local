from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.verification import VerificationIssue


class FinalVerificationReport(BaseModel):
    verdict: Literal["pass", "revise"]
    answer_complete: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def pass_requires_complete(self) -> "FinalVerificationReport":
        if self.verdict == "pass" and not self.answer_complete:
            raise ValueError("pass verdict requires answer_complete=true")
        return self
