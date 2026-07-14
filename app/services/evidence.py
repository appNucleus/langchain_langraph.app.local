from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.schemas.evidence import EvidenceItem

_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
_INJECTION_PATTERNS = (
    re.compile(r"ignore (?:(?:all|any|the|your) )?previous instructions", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"developer message", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"call (this|the) tool", re.I),
)


def canonicalize_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    try:
        parsed = urlsplit(uri.strip())
    except ValueError:
        return uri.strip() or None
    if not parsed.scheme or not parsed.netloc:
        return uri.strip() or None
    host = parsed.hostname.lower() if parsed.hostname else ""
    port = parsed.port
    if port and not (
        (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMETERS
        )
    )
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), host, path, query, ""))


def source_quality_for_uri(uri: str | None) -> str:
    if not uri:
        return "secondary_unknown"
    try:
        host = (urlsplit(uri).hostname or "").lower()
    except ValueError:
        return "secondary_unknown"
    if host.endswith(".gov") or host in {"who.int", "un.org"}:
        return "primary_authoritative"
    if host.endswith(".edu"):
        return "primary_non_authoritative"
    return "secondary_unknown"


def freshness_for_result(
    *,
    published_at: datetime | None,
    query: str,
    retrieved_at: datetime,
) -> str:
    lowered = query.lower()
    time_sensitive = any(
        token in lowered
        for token in (
            "current",
            "latest",
            "today",
            "tomorrow",
            "weather",
            "news",
            "score",
            "schedule",
            "price",
            "president",
            "ceo",
        )
    )
    if not time_sensitive:
        return "not_time_sensitive"
    if published_at is None:
        return "unknown"
    maximum_age = timedelta(days=1 if "weather" in lowered or "today" in lowered else 7)
    return "current" if retrieved_at - published_at <= maximum_age else "stale"


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_text(value: object, *, max_chars: int = 12000) -> tuple[str, bool]:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(text.replace("\x00", " ").split())
    truncated = len(text) > max_chars
    return text[:max_chars], truncated


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_for_prompt_injection(text: str) -> str:
    return (
        "suspicious"
        if any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
        else "clean"
    )


def evidence_from_metadata(
    metadata: dict[str, object],
    *,
    run_id: str = "request",
    task_id: str = "request",
) -> list[EvidenceItem]:
    """Read caller context without accepting caller-supplied trust labels."""

    raw = metadata.get("evidence", [])
    if not isinstance(raw, list):
        return []
    result: list[EvidenceItem] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            payload: dict[str, Any] = {"content": item}
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            payload = {"content": item}
        text, truncated = normalize_text(payload.get("content", payload))
        evidence_id = str(
            payload.get("id") or payload.get("evidence_id") or f"user-{index}"
        )
        result.append(
            EvidenceItem(
                evidence_id=evidence_id,
                run_id=run_id,
                task_id=task_id,
                source_title="request_metadata",
                normalized_text=text,
                summary=text,
                content_hash=content_hash(text),
                trust_class="user_supplied",
                freshness_status="unknown",
                source_quality="unknown",
                injection_scan_status=scan_for_prompt_injection(text),
                truncated=truncated,
                tool_status="not_applicable",
                eligible_for_claim_support=False,
                metadata={
                    key: value
                    for key, value in payload.items()
                    if key not in {"id", "evidence_id", "source", "content"}
                },
            )
        )
    return result


def retrieved_evidence(
    *,
    evidence_id: str,
    tool_name: str,
    raw_value: object,
    content: object,
    truncated: bool = False,
    run_id: str = "request",
    task_id: str = "request",
    query_id: str | None = None,
    query: str = "",
) -> EvidenceItem:
    """Build canonical evidence from one successful retrieved value.

    This established public helper preserves its compatibility signature while
    delegating to the canonical provenance and grounding fields.
    """

    normalized, normalized_truncated = normalize_text(content)
    source_uri = _first_uri(raw_value)
    canonical_uri = canonicalize_uri(source_uri)
    retrieved_at = datetime.now(UTC)
    published_at = _parse_datetime(
        _first_named_value(
            raw_value,
            ("published_at", "published", "date_published", "date"),
        )
    )
    source_title = _first_named_value(
        raw_value,
        ("title", "source_title", "name"),
    )
    content_type = _first_named_value(
        raw_value,
        ("content_type", "mime_type", "media_type"),
    )
    raw_artifact_uri = _first_named_value(
        raw_value,
        ("raw_artifact_uri", "artifact_uri", "object_uri"),
    )
    return EvidenceItem(
        evidence_id=evidence_id,
        run_id=run_id,
        task_id=task_id,
        query_id=query_id,
        tool_name=tool_name,
        source_uri=source_uri,
        canonical_uri=canonical_uri,
        source_title=str(source_title or tool_name),
        retrieved_at=retrieved_at,
        published_at=published_at,
        content_type=str(content_type or "text/plain"),
        raw_artifact_uri=str(raw_artifact_uri) if raw_artifact_uri else None,
        normalized_text=normalized,
        summary=normalized,
        content_hash=content_hash(normalized),
        trust_class="retrieved_external",
        freshness_status=freshness_for_result(
            published_at=published_at,
            query=query,
            retrieved_at=retrieved_at,
        ),
        source_quality=source_quality_for_uri(source_uri),
        injection_scan_status=scan_for_prompt_injection(normalized),
        truncated=bool(truncated or normalized_truncated),
        tool_status="success",
        eligible_for_claim_support=True,
        metadata={"query": query} if query else {},
    )


def evidence_from_tool_result(
    *,
    result: object,
    evidence_id: str,
    run_id: str,
    task_id: str,
    query_id: str | None,
    tool_name: str,
    query: str,
) -> EvidenceItem:
    ok = bool(getattr(result, "ok", False))
    data = getattr(result, "data", None)
    error = getattr(result, "error", None)
    payload = data if ok else error or "tool request failed"
    text, truncated = normalize_text(payload)
    source_uri = _first_uri(data)
    canonical_uri = canonicalize_uri(source_uri)
    retrieved_at = datetime.now(UTC)
    published_at = _parse_datetime(
        _first_named_value(
            data,
            ("published_at", "published", "date_published", "date"),
        )
    )
    source_title = _first_named_value(data, ("title", "source_title", "name"))
    raw_artifact_uri = _first_named_value(
        data,
        ("raw_artifact_uri", "artifact_uri", "object_uri"),
    )
    return EvidenceItem(
        evidence_id=evidence_id,
        run_id=run_id,
        task_id=task_id,
        query_id=query_id,
        tool_name=tool_name,
        source_uri=source_uri,
        canonical_uri=canonical_uri,
        source_title=str(source_title or tool_name),
        retrieved_at=retrieved_at,
        published_at=published_at,
        raw_artifact_uri=str(raw_artifact_uri) if raw_artifact_uri else None,
        normalized_text=text,
        summary=text,
        content_hash=content_hash(text),
        trust_class="retrieved_external" if ok else "tool_error",
        freshness_status=(
            freshness_for_result(
                published_at=published_at,
                query=query,
                retrieved_at=retrieved_at,
            )
            if ok
            else "unknown"
        ),
        source_quality=(source_quality_for_uri(source_uri) if ok else "unverifiable"),
        injection_scan_status=scan_for_prompt_injection(text),
        truncated=truncated,
        tool_status="success" if ok else "failed",
        eligible_for_claim_support=ok,
        metadata={"query": query},
    )


def deduplicate_evidence(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    output: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.canonical_uri or "", item.content_hash)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _first_named_value(value: object, keys: tuple[str, ...]) -> object | None:
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if candidate not in (None, ""):
                return candidate
        for nested in value.values():
            candidate = _first_named_value(nested, keys)
            if candidate not in (None, ""):
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = _first_named_value(nested, keys)
            if candidate not in (None, ""):
                return candidate
    return None


def _first_uri(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("url", "uri", "source_uri", "link"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for nested in value.values():
            candidate = _first_uri(nested)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = _first_uri(nested)
            if candidate:
                return candidate
    return None
