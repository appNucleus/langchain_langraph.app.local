from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Deque

from app.settings import Settings


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass
class _Session:
    messages: Deque[ConversationTurn]
    touched_at: float


class InMemoryConversationStore:
    """Bounded, TTL-based, process-local conversation memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._last_cleanup = 0.0

    async def get(self, thread_id: str) -> list[ConversationTurn]:
        async with self._lock:
            self._cleanup_locked()
            session = self._sessions.get(thread_id)
            if session is None:
                return []
            session.touched_at = monotonic()
            self._sessions.move_to_end(thread_id)
            return list(session.messages)

    async def add_pair(self, thread_id: str, *, user: str, assistant: str) -> None:
        async with self._lock:
            self._cleanup_locked()
            session = self._sessions.get(thread_id)
            if session is None:
                session = _Session(
                    messages=deque(maxlen=self.settings.session_history_messages),
                    touched_at=monotonic(),
                )
                self._sessions[thread_id] = session
            session.messages.append(ConversationTurn(role="user", content=self._trim(user)))
            session.messages.append(ConversationTurn(role="assistant", content=self._trim(assistant)))
            session.touched_at = monotonic()
            self._sessions.move_to_end(thread_id)
            while len(self._sessions) > self.settings.max_conversation_sessions:
                self._sessions.popitem(last=False)

    async def clear(self, thread_id: str) -> None:
        async with self._lock:
            self._sessions.pop(thread_id, None)

    def render(self, history: Sequence[ConversationTurn]) -> str:
        if not history:
            return "No prior messages in this session."
        lines: list[str] = []
        for item in history[-self.settings.session_history_messages :]:
            lines.append(f"{'User' if item.role == 'user' else 'Assistant'}: {self._trim(item.content)}")
        return "\n".join(lines)

    def _cleanup_locked(self) -> None:
        now = monotonic()
        if now - self._last_cleanup < self.settings.conversation_cleanup_interval_seconds:
            return
        cutoff = now - self.settings.conversation_session_ttl_seconds
        expired = [key for key, value in self._sessions.items() if value.touched_at < cutoff]
        for key in expired:
            self._sessions.pop(key, None)
        self._last_cleanup = now

    def _trim(self, text: str) -> str:
        compact = " ".join(str(text).split())
        limit = self.settings.max_history_message_chars
        return compact if len(compact) <= limit else compact[: max(0, limit - 20)].rstrip() + " …[trimmed]"
