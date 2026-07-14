from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    """Normalize a scalar string or a concrete string sequence.

    Local structured-output models occasionally return one string instead of a
    one-item list. Accept that deterministic shape, but reject mappings,
    nested containers, booleans, and other ambiguous values instead of silently
    stringifying them.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a string or a list of strings")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        normalized.append(item)
    return normalized


def _normalize_evidence_ids(value: Any) -> list[str]:
    """Normalize deterministic evidence-ID shapes without accepting ambiguity."""

    if value is None:
        return []
    if isinstance(value, bool):
        raise ValueError("evidence_ids cannot contain booleans")
    if isinstance(value, (str, int)):
        return [str(value)]
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise ValueError("evidence_ids must be an ID or a list of IDs")

    normalized: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(
                "evidence_ids entries must be non-boolean strings or integers"
            )
        normalized.append(str(item))
    return normalized


class Claim(BaseModel):
    # These fields support Stage 4 grounding. They remain absent from the
    # established public serialization when they carry only their defaults.
    # Stable fallback IDs are derived by the grounding service, not injected
    # into every WorkerResult returned to existing callers.
    claim_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    requires_current_evidence: bool = Field(
        default=False,
        exclude_if=lambda value: value is False,
    )

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> list[str]:
        return _normalize_evidence_ids(value)


class WorkerResult(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("assumptions", "missing_information", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any, info: Any) -> list[str]:
        return _normalize_string_list(value, field_name=info.field_name)
