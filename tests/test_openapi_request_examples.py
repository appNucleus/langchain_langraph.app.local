from __future__ import annotations

import json
from pathlib import Path

from app.schemas.chat import CHAT_REQUEST_EXAMPLE, ChatRequest


def test_default_openapi_request_is_minimal_and_runnable() -> None:
    assert CHAT_REQUEST_EXAMPLE == {"message": "Continue the analysis"}
    parsed = ChatRequest.model_validate(CHAT_REQUEST_EXAMPLE)
    assert parsed.message == "Continue the analysis"
    assert parsed.conversation_id is None
    assert parsed.run_id is None
    assert ChatRequest.model_json_schema()["examples"] == [CHAT_REQUEST_EXAMPLE]


def test_complete_request_contract_is_stored_separately() -> None:
    path = Path("docs/example_request/chat-complete.json")
    complete = json.loads(path.read_text(encoding="utf-8"))
    assert set(complete) == {
        "message",
        "thread_id",
        "conversation_id",
        "run_id",
        "resume",
        "resume_token",
        "system_prompt",
        "metadata",
    }
    parsed = ChatRequest.model_validate(complete)
    assert parsed.message == "Continue the analysis"
