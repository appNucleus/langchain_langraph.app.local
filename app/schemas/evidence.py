from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TrustClass = Literal[
    "retrieved_external",
    "user_supplied",
    "internal_system",
    "derived_summary",
    "tool_error",
]
FreshnessStatus = Literal["current", "stale", "unknown", "not_applicable"]
SourceQuality = Literal["official", "high", "medium", "low", "unknown"]


class EvidenceItem(BaseModel):
    """Typed evidence record with backward-compatible provenance metadata."""

    id: str
    source: str
    content: str
    run_id: str | None = None
    task_id: str | None = None
    query_id: str | None = None
    tool_name: str | None = None
    trust_class: TrustClass = "retrieved_external"
    source_uri: str | None = None
    source_title: str | None = None
    retrieved_at: datetime | None = None
    published_at: datetime | None = None
    content_type: str = "text/plain"
    content_hash: str = ""
    freshness_status: FreshnessStatus = "unknown"
    source_quality: SourceQuality = "unknown"
    truncated: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("source_uri", "source_title", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def normalize_provenance(self) -> "EvidenceItem":
        if self.source == "request_metadata":
            self.trust_class = "user_supplied"
            self.source_quality = "unknown"
        if not self.content_hash:
            self.content_hash = sha256(self.content.encode("utf-8")).hexdigest()
        return self

    def prompt_record(self, *, content: str | None = None) -> dict[str, object]:
        """Return the bounded record exposed to worker/verifier prompts."""

        return {
            "id": self.id,
            "source": self.source,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "query_id": self.query_id,
            "tool_name": self.tool_name,
            "source_uri": self.source_uri,
            "source_title": self.source_title,
            "trust_class": self.trust_class,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "content_type": self.content_type,
            "content_hash": self.content_hash,
            "freshness_status": self.freshness_status,
            "source_quality": self.source_quality,
            "truncated": self.truncated,
            "content": self.content if content is None else content,
            "metadata": dict(self.metadata),
        }
