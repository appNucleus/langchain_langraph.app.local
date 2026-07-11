from __future__ import annotations

def deterministic_output_issues(text: str, minimum_chars: int = 20) -> list[str]:
    issues: list[str] = []
    clean = text.strip()
    if len(clean) < minimum_chars:
        issues.append('answer_too_short')
    if clean.count(clean[:50]) > 3 and len(clean) > 100:
        issues.append('possible_repetition')
    return issues
