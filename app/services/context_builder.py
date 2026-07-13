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
    """Build bounded, deduplicated evidence with explicit provenance fields."""

    remaining = max(0, int(max_chars))
    per_item = max(1, int(max_item_chars or max_chars or 1))
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        if remaining <= 0:
            break
        key = (item.id, item.content_hash)
        if key in seen:
            continue
        seen.add(key)

        allowed = min(remaining, per_item)
        content = item.content[:allowed]
        context_truncated = len(item.content) > len(content)
        record = item.prompt_record(content=content)
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("original_content_chars", len(item.content))
        metadata["context_content_chars"] = len(content)
        metadata["context_truncated"] = context_truncated
        record["metadata"] = metadata
        record["truncated"] = bool(item.truncated or context_truncated)
        output.append(record)
        remaining -= len(content)

    return output


def context_character_count(items: Iterable[dict[str, Any]]) -> int:
    return sum(len(str(item.get("content") or "")) for item in items)
