from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConversationStore(ABC):
    """Conversation-history abstraction used by the graph runtime."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize external resources."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release external resources."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return a small readiness payload."""

    @abstractmethod
    async def get(self, thread_id: str) -> list[dict[str, Any]]:
        """Return conversation messages in chronological order."""

    @abstractmethod
    async def append(self, thread_id: str, *messages: dict[str, Any]) -> None:
        """Append one or more messages atomically where supported."""

    @abstractmethod
    async def clear(self, thread_id: str) -> None:
        """Delete one conversation thread."""
