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
    """Build a bounded, deduplicated evidence context.

    The previous implementation could place one very large MCP result into the
    worker prompt until the entire character allowance was consumed. Combined
    with the JSON output schema, this could fill Ollama's context window and
    leave only one generation token. This function now bounds both the aggregate
    and each individual evidence item while preserving stable evidence IDs.
    """

    remaining = max(0, int(max_chars))
    per_item = max(1, int(max_item_chars or max_chars or 1))
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        if remaining <= 0:
            break

        key = (item.id, item.content)
        if key in seen:
            continue
        seen.add(key)

        allowed = min(remaining, per_item)
        original = item.content
        content = original[:allowed]
        truncated = len(original) > len(content)
        metadata = dict(item.metadata)
        metadata.setdefault("original_content_chars", len(original))
        metadata["context_content_chars"] = len(content)
        metadata["context_truncated"] = truncated

        output.append(
            {
                "id": item.id,
                "source": item.source,
                "content": content,
                "metadata": metadata,
            }
        )
        remaining -= len(content)

    return output


def context_character_count(items: Iterable[dict[str, Any]]) -> int:
    """Return only evidence-content characters, without serializing bodies."""

    return sum(len(str(item.get("content") or "")) for item in items)
