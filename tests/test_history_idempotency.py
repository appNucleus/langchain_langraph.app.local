from __future__ import annotations

import pytest

from app.state.in_memory import BoundedInMemoryStore


@pytest.mark.asyncio
async def test_memory_history_appends_each_run_once_after_visible_trimming() -> None:
    store = BoundedInMemoryStore(ttl_seconds=60, max_sessions=10, max_messages=2)
    user = {"role": "user", "content": "hello"}
    assistant = {"role": "assistant", "content": "hi"}

    assert await store.append_turn(
        "conversation",
        run_id="run-1",
        user_message=user,
        assistant_message=assistant,
    )
    assert not await store.append_turn(
        "conversation",
        run_id="run-1",
        user_message=user,
        assistant_message=assistant,
    )
    assert len(await store.get("conversation")) == 2

    assert await store.append_turn(
        "conversation",
        run_id="run-2",
        user_message=user,
        assistant_message=assistant,
    )
    assert len(await store.get("conversation")) == 2
    assert not await store.append_turn(
        "conversation",
        run_id="run-1",
        user_message=user,
        assistant_message=assistant,
    )
