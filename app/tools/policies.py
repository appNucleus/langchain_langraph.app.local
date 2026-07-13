from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class SideEffectLevel(StrEnum):
    READ_ONLY = "read_only"
    EXTERNAL_READ = "external_read"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_COMMUNICATION = "external_communication"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class ToolPolicy:
    level: SideEffectLevel
    known: bool = True
    cacheable: bool = False

    @property
    def read_only(self) -> bool:
        return self.level in {SideEffectLevel.READ_ONLY, SideEffectLevel.EXTERNAL_READ}


POLICIES: dict[str, ToolPolicy] = {
    "mail_search": ToolPolicy(SideEffectLevel.EXTERNAL_READ),
    "mail_create_draft": ToolPolicy(SideEffectLevel.REVERSIBLE_WRITE),
    "mail_send_draft": ToolPolicy(SideEffectLevel.EXTERNAL_COMMUNICATION),
}

_WRITE_TOKENS = {
    "create",
    "update",
    "delete",
    "remove",
    "send",
    "publish",
    "post",
    "write",
    "modify",
    "forward",
    "reply",
    "book",
    "purchase",
    "order",
    "upload",
    "put",
    "patch",
}
_READ_TOKENS = {
    "search",
    "read",
    "get",
    "list",
    "find",
    "lookup",
    "query",
    "fetch",
    "scrape",
    "news",
    "weather",
    "status",
    "inventory",
    "inspect",
    "describe",
}


def policy_for(tool: str) -> ToolPolicy:
    """Resolve a fail-closed policy for explicit and dynamically discovered tools."""

    normalized = re.sub(r"[^a-z0-9]+", "_", tool.strip().lower()).strip("_")
    explicit = POLICIES.get(normalized)
    if explicit is not None:
        return explicit

    tokens = {token for token in normalized.split("_") if token}
    if tokens & _WRITE_TOKENS:
        return ToolPolicy(SideEffectLevel.RESTRICTED, known=False)
    if tokens & _READ_TOKENS:
        return ToolPolicy(SideEffectLevel.EXTERNAL_READ, known=False)
    return ToolPolicy(SideEffectLevel.RESTRICTED, known=False)
