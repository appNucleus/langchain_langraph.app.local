from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceItem
from app.schemas.worker import Claim

GroundingStatus = Literal[
    "supported",
    "partially_supported",
    "contradicted",
    "stale",
    "unsupported",
]


class ClaimGroundingResult(BaseModel):
    claim_id: str
    status: GroundingStatus
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str


def ground_claims(
    claims: list[Claim],
    evidence: list[EvidenceItem],
    *,
    run_id: str,
    task_id: str,
) -> list[ClaimGroundingResult]:
    by_id = {item.evidence_id: item for item in evidence}
    results: list[ClaimGroundingResult] = []
    for index, claim in enumerate(claims, start=1):
        claim_id = claim.claim_id or f"claim-{index}"
        requested = [by_id.get(evidence_id) for evidence_id in claim.evidence_ids]
        valid = [
            item
            for item in requested
            if item is not None
            and item.run_id == run_id
            and item.task_id == task_id
            and item.eligible_for_claim_support
            and item.tool_status == "success"
        ]
        stale = [item for item in valid if item.freshness_status == "stale"]
        acceptable = [
            item
            for item in valid
            if (
                item.freshness_status == "current"
                if claim.requires_current_evidence
                else item.freshness_status != "stale"
            )
        ]
        if claim.requires_current_evidence and stale and not acceptable:
            status: GroundingStatus = "stale"
            explanation = "All otherwise valid supporting evidence is stale or not current."
        elif acceptable and len(acceptable) == len(claim.evidence_ids):
            status = "supported"
            explanation = "Every referenced evidence record is valid for this run and task."
        elif acceptable:
            status = "partially_supported"
            explanation = "Some referenced evidence is missing, ineligible, stale, or foreign."
        else:
            status = "unsupported"
            explanation = "No eligible same-run, same-task evidence supports the claim."
        results.append(
            ClaimGroundingResult(
                claim_id=claim_id,
                status=status,
                supporting_evidence_ids=[item.evidence_id for item in acceptable],
                explanation=explanation,
            )
        )
    return results
