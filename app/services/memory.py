from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Deque

from app.settings import Settings


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


class InMemoryConversationStore:
    """Small thread-aware memory store for local single-process use.

    This is intentionally not a database. It gives the current session enough
    concise context without adding infrastructure complexity. Replace it later
    with Redis/Postgres checkpointers if multi-process durability is required.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._messages: dict[str, Deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=settings.session_history_messages)
        )

    async def get(self, thread_id: str) -> list[ConversationTurn]:
        async with self._lock:
            return list(self._messages[thread_id])

    async def add_pair(self, thread_id: str, *, user: str, assistant: str) -> None:
        async with self._lock:
            self._messages[thread_id].append(
                ConversationTurn(role="user", content=self._trim(user))
            )
            self._messages[thread_id].append(
                ConversationTurn(role="assistant", content=self._trim(assistant))
            )

    def render(self, history: Sequence[ConversationTurn]) -> str:
        if not history:
            return "No prior messages in this session."
        lines: list[str] = []
        for item in history[-self.settings.session_history_messages :]:
            role = "User" if item.role == "user" else "Assistant"
            lines.append(f"{role}: {self._trim(item.content)}")
        return "\n".join(lines)

    def _trim(self, text: str) -> str:
        text = " ".join(str(text).split())
        limit = self.settings.max_history_message_chars
        if len(text) <= limit:
            return text
        return text[: limit - 20].rstrip() + " …[trimmed]"
