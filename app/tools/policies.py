from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class SideEffectLevel(StrEnum):
    READ_ONLY='read_only'; EXTERNAL_READ='external_read'; REVERSIBLE_WRITE='reversible_write'; EXTERNAL_COMMUNICATION='external_communication'; RESTRICTED='restricted'
@dataclass(frozen=True)
class ToolPolicy:
    level: SideEffectLevel=SideEffectLevel.READ_ONLY
    confirmation_required: bool=False
    cacheable: bool=False

POLICIES={
 'mail_search':ToolPolicy(SideEffectLevel.EXTERNAL_READ),
 'mail_create_draft':ToolPolicy(SideEffectLevel.REVERSIBLE_WRITE, confirmation_required=True),
 'mail_send_draft':ToolPolicy(SideEffectLevel.EXTERNAL_COMMUNICATION, confirmation_required=True),
}
def policy_for(tool: str) -> ToolPolicy: return POLICIES.get(tool,ToolPolicy())
