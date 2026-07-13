from __future__ import annotations

from app.settings import Settings
from app.state.runtime import StateRuntime


class BrokenCheckpointer:
    async def aget_tuple(self, _config):
        raise RuntimeError("checkpoint database unavailable")


async def test_memory_checkpointer_health_is_reported() -> None:
    runtime = StateRuntime(Settings(llm_backend="echo"))
    await runtime.start()
    try:
        health = await runtime.health()
    finally:
        await runtime.aclose()

    assert health["checkpoint"]["status"] == "available"
    assert health["checkpoint"]["backend"] == "memory"


async def test_broken_checkpointer_is_reported_unavailable() -> None:
    runtime = StateRuntime(
        Settings(llm_backend="echo", expose_internal_health_details=True)
    )
    runtime.checkpointer = BrokenCheckpointer()

    health = await runtime.health()

    assert health["checkpoint"]["status"] == "unavailable"
    assert "checkpoint database unavailable" in health["checkpoint"]["error"]
