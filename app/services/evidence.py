from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.schemas.evidence import EvidenceItem

_SOURCE_URI_KEYS = ("url", "source_url", "uri", "link", "href")
_SOURCE_TITLE_KEYS = ("title", "source_title", "name", "headline")
_PUBLISHED_KEYS = ("published_at", "published", "date", "published_date")




def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None

def _first_scalar(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float)) and str(candidate).strip():
                return str(candidate).strip()
        for candidate in value.values():
            found = _first_scalar(candidate, keys)
            if found:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _first_scalar(candidate, keys)
            if found:
                return found
    return None


def extract_tool_provenance(value: Any) -> dict[str, object]:
    """Extract conservative source metadata without claiming unsupported quality."""

    published_at = _first_scalar(value, _PUBLISHED_KEYS)
    return {
        "source_uri": _first_scalar(value, _SOURCE_URI_KEYS),
        "source_title": _first_scalar(value, _SOURCE_TITLE_KEYS),
        "published_at": published_at,
        "content_type": "application/json" if not isinstance(value, str) else "text/plain",
    }


def evidence_from_metadata(metadata: dict[str, object]) -> list[EvidenceItem]:
    """Convert caller evidence to explicitly untrusted user-supplied records."""

    raw = metadata.get("evidence", [])
    if not isinstance(raw, list):
        return []

    result: list[EvidenceItem] = []
    run_id = str(metadata.get("run_id") or "").strip() or None
    for index, item in enumerate(raw):
        if isinstance(item, str):
            result.append(
                EvidenceItem(
                    id=f"e{index + 1}",
                    source="request_metadata",
                    content=item,
                    run_id=run_id,
                    trust_class="user_supplied",
                    freshness_status="unknown",
                    source_quality="unknown",
                )
            )
            continue
        if not isinstance(item, dict):
            continue

        excluded = {
            "id",
            "source",
            "content",
            "trust_class",
            "source_uri",
            "source_title",
            "retrieved_at",
            "published_at",
            "content_type",
            "content_hash",
            "freshness_status",
            "source_quality",
            "truncated",
        }
        result.append(
            EvidenceItem(
                id=str(item.get("id") or f"e{index + 1}"),
                source=str(item.get("source") or "request_metadata"),
                content=str(
                    item.get("content")
                    or json.dumps(item, ensure_ascii=False, default=str)
                ),
                run_id=run_id,
                task_id=str(item.get("task_id") or "").strip() or None,
                query_id=str(item.get("query_id") or "").strip() or None,
                tool_name=None,
                # Caller input cannot promote itself to independently retrieved evidence.
                trust_class="user_supplied",
                source_uri=item.get("source_uri") or item.get("url"),
                source_title=item.get("source_title") or item.get("title"),
                retrieved_at=_parse_datetime(item.get("retrieved_at")),
                published_at=_parse_datetime(item.get("published_at")),
                content_type=str(item.get("content_type") or "text/plain"),
                freshness_status="unknown",
                source_quality="unknown",
                truncated=bool(item.get("truncated", False)),
                metadata={key: value for key, value in item.items() if key not in excluded},
            )
        )
    return result


def retrieved_evidence(
    *,
    evidence_id: str,
    tool_name: str,
    raw_value: Any,
    run_id: str | None = None,
    task_id: str | None = None,
    query_id: str | None = None,
    content: str,
    truncated: bool,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    provenance = extract_tool_provenance(raw_value)
    return EvidenceItem(
        id=evidence_id,
        source=tool_name,
        content=content,
        run_id=run_id,
        task_id=task_id,
        query_id=query_id,
        tool_name=tool_name,
        trust_class="retrieved_external",
        source_uri=provenance.get("source_uri"),
        source_title=provenance.get("source_title"),
        retrieved_at=datetime.now(UTC),
        published_at=_parse_datetime(provenance.get("published_at")),
        content_type=str(provenance.get("content_type") or "text/plain"),
        freshness_status="unknown",
        source_quality="unknown",
        truncated=truncated,
        metadata=dict(metadata or {}),
    )
