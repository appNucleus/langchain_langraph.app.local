from __future__ import annotations
import json
from app.schemas.evidence import EvidenceItem

def evidence_from_metadata(metadata: dict[str, object]) -> list[EvidenceItem]:
    raw = metadata.get('evidence', [])
    if not isinstance(raw, list):
        return []
    result: list[EvidenceItem] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            result.append(EvidenceItem(id=f'e{index+1}', source='request_metadata', content=item))
        elif isinstance(item, dict):
            result.append(EvidenceItem(
                id=str(item.get('id') or f'e{index+1}'),
                source=str(item.get('source') or 'request_metadata'),
                content=str(item.get('content') or json.dumps(item, ensure_ascii=False)),
                metadata={k: v for k, v in item.items() if k not in {'id','source','content'}},
            ))
    return result
