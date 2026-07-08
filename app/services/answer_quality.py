from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


BAD_EXACT = {
    "",
    "**",
    "***",
    "-",
    "--",
    "1",
    "0",
    "yes",
    "no",
    "none",
    "null",
    "no answer content returned.",
}


@dataclass(frozen=True)
class AnswerValidation:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": self.reasons}


def validate_answer(answer: str, *, query: str, min_chars: int = 80, required_references: bool = False, references_count: int = 0) -> AnswerValidation:
    """Cheap deterministic validation for subquery answers.

    It intentionally does not judge truth. It only blocks obvious model/tool
    failures, such as markdown fragments, one-character outputs, empty content,
    or source-required answers with no source evidence.
    """
    cleaned = _normalize(answer)
    reasons: list[str] = []

    if cleaned.lower() in BAD_EXACT:
        reasons.append("answer_is_empty_or_placeholder")
    if len(cleaned) < min_chars:
        reasons.append("answer_too_short")
    if not re.search(r"[A-Za-z0-9]", cleaned):
        reasons.append("answer_has_no_alphanumeric_content")
    if _mostly_markup_or_numbers(cleaned):
        reasons.append("answer_is_mostly_markup_or_numbers")
    if required_references and references_count <= 0:
        reasons.append("references_required_but_missing")

    # Truth/relevance is judged by the model/prompt; this gate only blocks obvious transport/model failures.


    return AnswerValidation(ok=not reasons, reasons=reasons)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _mostly_markup_or_numbers(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    useful = re.findall(r"[A-Za-z]", compact)
    if len(useful) >= 12:
        return False
    markup_or_number = re.findall(r"[`*_#>\-+={}\[\]().,;:!?/\\|0-9]", compact)
    return len(markup_or_number) / max(len(compact), 1) > 0.65


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "does",
    "are",
    "use",
    "using",
    "into",
    "from",
    "then",
    "each",
    "app",
    "task",
}
