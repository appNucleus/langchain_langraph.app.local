from __future__ import annotations
from app.schemas.evidence import EvidenceItem

def build_context(items: list[EvidenceItem], max_chars: int) -> list[dict[str, object]]:
    remaining = max_chars
    output: list[dict[str, object]] = []
    for item in items:
        if remaining <= 0:
            break
        content = item.content[:remaining]
        output.append({'id': item.id, 'source': item.source, 'content': content, 'metadata': item.metadata})
        remaining -= len(content)
    return output
