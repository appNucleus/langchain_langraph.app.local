from __future__ import annotations
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    id: str
    source: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
