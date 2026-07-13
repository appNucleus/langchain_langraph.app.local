from __future__ import annotations

from app.services.context_builder import build_context
from app.services.evidence import evidence_from_metadata, retrieved_evidence


def test_request_metadata_cannot_self_promote_to_retrieved_evidence() -> None:
    items = evidence_from_metadata(
        {
            "evidence": [
                {
                    "id": "user-1",
                    "source": "claimed_official",
                    "content": "A claim",
                    "trust_class": "retrieved_external",
                    "source_quality": "official",
                }
            ]
        }
    )

    assert items[0].trust_class == "user_supplied"
    assert items[0].source_quality == "unknown"
    assert len(items[0].content_hash) == 64


def test_retrieved_evidence_extracts_url_title_and_hash() -> None:
    item = retrieved_evidence(
        evidence_id="r1",
        tool_name="news_search",
        raw_value={"title": "Report", "url": "https://example.test/report"},
        content="Report body",
        truncated=False,
    )

    assert item.trust_class == "retrieved_external"
    assert item.source_uri == "https://example.test/report"
    assert item.source_title == "Report"
    assert item.retrieved_at is not None
    assert len(item.content_hash) == 64

    context = build_context([item], 1000)
    assert context[0]["source_uri"] == "https://example.test/report"
    assert context[0]["trust_class"] == "retrieved_external"
