from __future__ import annotations

from app.schemas.chat import ChatRequest
from app.schemas.evidence import EvidenceItem
from app.services.context_builder import build_context, context_character_count
from app.services.routing import RuntimeRouter
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        ollama_num_ctx=8192,
        research_max_queries_per_task=3,
    )


def test_blank_system_prompt_is_optional_and_derived_from_message() -> None:
    request = ChatRequest(
        message="Find yesterday's FIFA match involving Argentina and analyze the red card",
        system_prompt="   ",
    )
    assert request.system_prompt is None

    decision = RuntimeRouter(_settings()).prepare_system_prompt(
        message=request.message,
        provided=request.system_prompt,
    )
    assert decision.source == "derived"
    assert decision.domain == "sports"
    assert "sports research analyst" in decision.prompt
    assert "external evidence" in decision.prompt
    assert decision.requires_external_evidence is True


def test_nonblank_system_prompt_is_preserved() -> None:
    decision = RuntimeRouter(_settings()).prepare_system_prompt(
        message="Analyze a stable architecture tradeoff",
        provided="Use my exact specialist instruction.",
    )
    assert decision.source == "request"
    assert decision.prompt == "Use my exact specialist instruction."


def test_compound_research_task_creates_multiple_bounded_queries() -> None:
    router = RuntimeRouter(_settings())
    queries = router.build_research_queries(
        user_request=(
            "Find yesterday's FIFA World Cup game involving Argentina and analyze "
            "criticism of the red-card decision"
        ),
        task={
            "objective": "Identify the correct match and red-card incident",
            "required_evidence": [
                "official match report and date",
                "reputable criticism of the referee decision",
            ],
        },
        required_actions=["Find an independent rules analysis"],
        limit=3,
    )

    assert len(queries) == 3
    assert len({query.casefold() for query in queries}) == 3
    assert all(len(query) <= 800 for query in queries)
    assert any("official match report" in query.lower() for query in queries)
    assert all("event date" in query for query in queries)


def test_context_builder_bounds_each_item_and_total() -> None:
    items = [
        EvidenceItem(id="e1", source="tool", content="a" * 9000),
        EvidenceItem(id="e2", source="tool", content="b" * 9000),
    ]
    context = build_context(items, 7000, max_item_chars=4000)

    assert len(context) == 2
    assert len(context[0]["content"]) == 4000
    assert len(context[1]["content"]) == 3000
    assert context_character_count(context) == 7000
    assert context[0]["metadata"]["context_truncated"] is True
