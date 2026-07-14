from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.schemas.evidence import EvidenceItem


def build_context(
    items: list[EvidenceItem],
    max_chars: int,
    *,
    max_item_chars: int | None = None,
) -> list[dict[str, object]]:
    """Build a bounded, deduplicated context with canonical provenance.

    The returned dictionaries retain the established ``id``, ``source``,
    ``content`` and ``metadata`` keys while exposing the newer Stage 4 fields.
    Retrieved text remains explicitly delimited as untrusted data.
    """

    remaining = max(0, int(max_chars))
    per_item = max(1, int(max_item_chars or max_chars or 1))
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        if remaining <= 0:
            break
        key = (item.evidence_id, item.normalized_text)
        if key in seen:
            continue
        seen.add(key)

        allowed = min(remaining, per_item)
        original = item.normalized_text
        body = original[:allowed]
        context_truncated = item.truncated or len(original) > len(body)
        metadata = dict(item.metadata)
        metadata.setdefault("original_content_chars", len(original))
        metadata["context_content_chars"] = len(body)
        metadata["context_truncated"] = context_truncated
        wrapped = (
            f'<untrusted_evidence evidence_id="{item.evidence_id}">'
            f"{body}"
            "</untrusted_evidence>"
        )
        output.append(
            {
                "id": item.evidence_id,
                "evidence_id": item.evidence_id,
                "source": item.source,
                "source_uri": item.source_uri,
                "canonical_uri": item.canonical_uri,
                "source_title": item.source_title,
                "trust_class": item.trust_class,
                "freshness_status": item.freshness_status,
                "source_quality": item.source_quality,
                "tool_status": item.tool_status,
                "eligible_for_claim_support": item.eligible_for_claim_support,
                "content": wrapped,
                "metadata": metadata,
            }
        )
        remaining -= len(body)
    return output


def context_character_count(items: Iterable[dict[str, Any]]) -> int:
    """Return evidence-body characters without counting wrapper markup."""

    total = 0
    for item in items:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            measured = metadata.get("context_content_chars")
            if isinstance(measured, int) and measured >= 0:
                total += measured
                continue
        content = str(item.get("content") or "")
        opening_end = content.find(">")
        closing_start = content.rfind("</untrusted_evidence>")
        if content.startswith("<untrusted_evidence ") and 0 <= opening_end < closing_start:
            content = content[opening_end + 1 : closing_start]
        total += len(content)
    return total
