from __future__ import annotations

from app.schemas.evidence import EvidenceItem


def build_context(items: list[EvidenceItem], max_chars: int) -> list[dict[str, object]]:
    remaining = max(0, int(max_chars))
    output: list[dict[str, object]] = []
    for item in items:
        if remaining <= 0:
            break
        content = item.normalized_text[:remaining]
        output.append(
            {
                "evidence_id": item.evidence_id,
                "source": item.source,
                "trust_class": item.trust_class,
                "freshness_status": item.freshness_status,
                "source_quality": item.source_quality,
                "tool_status": item.tool_status,
                "eligible_for_claim_support": item.eligible_for_claim_support,
                "content": (
                    f'<untrusted_evidence evidence_id="{item.evidence_id}">'
                    f"{content}"
                    "</untrusted_evidence>"
                ),
                "metadata": item.metadata,
            }
        )
        remaining -= len(content)
    return output
