from __future__ import annotations

import pytest

from app.state.in_memory import BoundedInMemoryStore


@pytest.mark.asyncio
async def test_store_limits_messages_and_evicts_least_recently_used_session() -> None:
    store = BoundedInMemoryStore(
        ttl_seconds=60,
        max_sessions=2,
        max_messages=2,
    )

    await store.append("a", {"content": "1"}, {"content": "2"}, {"content": "3"})
    await store.append("b", {"content": "b"})

    assert [item["content"] for item in await store.get("a")] == ["2", "3"]

    await store.append("c", {"content": "c"})

    assert await store.get("b") == []
    assert (await store.get("a"))[0]["content"] == "2"
    assert (await store.get("c"))[0]["content"] == "c"


@pytest.mark.asyncio
async def test_store_ttl_tracks_session_inactivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr("app.state.in_memory.monotonic", lambda: now)
    store = BoundedInMemoryStore(
        ttl_seconds=10,
        max_sessions=2,
        max_messages=2,
    )
    await store.append("thread", {"content": "x"})

    now = 109.0
    assert await store.get("thread")

    now = 118.0
    assert await store.get("thread")

    now = 129.0
    assert await store.get("thread") == []
