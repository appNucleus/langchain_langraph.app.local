from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    claim_id: str | None = None
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

    uncertainty: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    requires_current_evidence: bool = Field(
        default=False,
        exclude_if=lambda value: value is False,
    )


class WorkerResult(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def assign_stable_claim_ids(self) -> "WorkerResult":
        for index, claim in enumerate(self.claims, start=1):
            if not claim.claim_id:
                claim.claim_id = f"claim-{index}"
        return self
