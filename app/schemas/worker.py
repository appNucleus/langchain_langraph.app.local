from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_string_list(
    value: Any,
    *,
    allow_integer_items: bool = False,
) -> Any:
    """Normalize common local-model JSON variants before strict validation.

    The canonical schema remains ``list[str]``. This helper only repairs
    unambiguous representation differences:

    * ``null`` becomes an empty list;
    * a single string becomes a one-item list;
    * integer identifiers may be converted to strings when explicitly allowed;
    * list items otherwise remain subject to normal Pydantic validation.

    It deliberately does not map numeric evidence references to evidence by
    position. An identifier such as ``0`` is preserved as ``"0"`` so the
    verifier can reject it when no evidence item actually has that ID.
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if allow_integer_items and isinstance(value, int) and not isinstance(value, bool):
        return [str(value)]

    if not isinstance(value, list):
        return value

    if not allow_integer_items:
        return value

    normalized: list[Any] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            normalized.append(str(item))
        else:
            normalized.append(item)
    return normalized


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        """Accept numeric IDs emitted by local models without weakening IDs."""

        return _normalize_string_list(value, allow_integer_items=True)


class WorkerResult(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("assumptions", "missing_information", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> Any:
        """Normalize singleton text emitted instead of a JSON string array."""

        return _normalize_string_list(value)
