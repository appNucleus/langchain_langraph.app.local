from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

TrustClass = Literal[
    "retrieved_external",
    "user_supplied",
    "internal_system",
    "derived_summary",
    "tool_error",
    "unknown_legacy",
]
FreshnessStatus = Literal["current", "stale", "unknown", "not_time_sensitive"]
SourceQuality = Literal[
    "primary_authoritative",
    "primary_non_authoritative",
    "secondary_reputable",
    "secondary_unknown",
    "user_supplied",
    "unverifiable",
    "unknown",
]
InjectionScanStatus = Literal["not_scanned", "clean", "suspicious", "blocked"]
ToolStatus = Literal["success", "failed", "timeout", "not_applicable", "unknown"]


class EvidenceItem(BaseModel):
    """Canonical evidence record with a conservative legacy reader."""

    model_config = ConfigDict(populate_by_name=True)

    evidence_id: str = Field(validation_alias=AliasChoices("evidence_id", "id"))
    run_id: str = "legacy"
    task_id: str = "legacy"
    query_id: str | None = None
    tool_name: str | None = None
    source_uri: str | None = None
    canonical_uri: str | None = None
    source_title: str | None = Field(
        default=None, validation_alias=AliasChoices("source_title", "source")
    )
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    published_at: datetime | None = None
    content_type: str = "text/plain"
    raw_artifact_uri: str | None = None
    normalized_text: str = Field(
        default="", validation_alias=AliasChoices("normalized_text", "content")
    )
    summary: str = ""
    content_hash: str = ""
    trust_class: TrustClass = "unknown_legacy"
    freshness_status: FreshnessStatus = "unknown"
    source_quality: SourceQuality = "unknown"
    injection_scan_status: InjectionScanStatus = "not_scanned"
    truncated: bool = False
    tool_status: ToolStatus = "unknown"
    eligible_for_claim_support: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_non_supporting_failures(self) -> "EvidenceItem":
        if not self.summary:
            self.summary = self.normalized_text
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.normalized_text.encode("utf-8")
            ).hexdigest()
        if self.trust_class in {"tool_error", "user_supplied", "unknown_legacy"}:
            self.eligible_for_claim_support = False
        if self.tool_status in {"failed", "timeout"}:
            self.eligible_for_claim_support = False
        return self

    @property
    def id(self) -> str:
        return self.evidence_id

    @property
    def source(self) -> str:
        return self.source_title or self.tool_name or self.source_uri or "unknown"

    @property
    def content(self) -> str:
        return self.normalized_text
